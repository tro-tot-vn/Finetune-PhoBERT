import os
import pandas as pd
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
from callbacks import JsonLossLogger
from report_generator import generate_report, load_training_history

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate

# ========================== CẤU HÌNH CƠ BẢN ==========================
MODEL_NAME = "vinai/phobert-base-v2"
DATA_DIR   = "data"      # chứa train.csv, valid.csv, test.csv
OUTPUT_DIR = "out/phobert-moderation"
SAVE_DIR   = "models/phobert-moderation"

MAX_LENGTH = 192
BATCH_TRAIN = 48         # ↑↑ TĂNG MẠNH để ăn đầy VRAM! (was 24, too small!)
BATCH_EVAL  = 96         # ↑↑ TĂNG MẠNH (was 48)
GRAD_ACCUM  = 1          # Giữ nguyên
EPOCHS      = 2          # Optimal cho large dataset >500K
LR          = 2e-5

SEED        = 42

# GPU Optimization flags
USE_GRADIENT_CHECKPOINTING = False  # ❌ TẮT để ăn nhiều VRAM hơn, train nhanh hơn!
USE_TORCH_COMPILE = False           # PyTorch 2.x compile (có thể làm chậm trên Colab)
# ====================================================================

def read_split(data_dir):
    def _load(path):
        df = pd.read_csv(path)
        # chuẩn hoá kiểu dữ liệu
        df["text"] = df["text"].astype(str)
        df["label"] = df["label"].astype(int)
        return df

    train_df = _load(os.path.join(data_dir, "train.csv"))
    valid_df = _load(os.path.join(data_dir, "valid.csv"))
    test_df  = _load(os.path.join(data_dir, "test.csv"))

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(valid_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })
    return ds, train_df, valid_df, test_df

def build_tokenizer():
    # PhoBERT thường dùng tokenizer không-fast
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    return tok

def encode_datasets(ds, tokenizer):
    def preprocess(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
    enc = ds.map(preprocess, batched=True, remove_columns=["text"])
    return enc

# ====== Metrics (ưu tiên lớp invalid = 1) ======
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits[:, 1] > logits[:, 0]).astype(int)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision_inv": precision.compute(predictions=preds, references=labels, pos_label=1)["precision"],
        "recall_inv": recall.compute(predictions=preds, references=labels, pos_label=1)["recall"],
        "f1_inv": f1.compute(predictions=preds, references=labels, pos_label=1)["f1"],
    }

# ====== Tự động class weights nếu lệch lớp ======
def get_class_weights(train_df):
    labels = train_df["label"].values
    classes = np.unique(labels)
    if len(classes) == 2:
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
        # trả về theo thứ tự [w_class0, w_class1]
        return torch.tensor(weights, dtype=torch.float)
    return None

class WeightedTrainer(Trainer):
    def __init__(self, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with class weights.
        Compatible with transformers>=4.35.0 (added num_items_in_batch parameter).
        """
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        if labels is not None:
            if self.class_weight is not None:
                loss_fct = CrossEntropyLoss(weight=self.class_weight.to(logits.device))
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        else:
            loss = None
        return (loss, outputs) if return_outputs else loss

def get_data_stats(train_df, valid_df, test_df):
    """Tính thống kê data cho báo cáo."""
    stats = {}
    for name, df in [("train", train_df), ("validation", valid_df), ("test", test_df)]:
        total = len(df)
        class_0 = (df["label"] == 0).sum()
        class_1 = (df["label"] == 1).sum()
        stats[name] = {
            "total": int(total),
            "class_0": int(class_0),
            "class_0_pct": float(class_0 / total * 100),
            "class_1": int(class_1),
            "class_1_pct": float(class_1 / total * 100),
        }
    return stats

def main():
    # Fix tqdm for Colab/Jupyter - prevent progress bar spam
    import sys
    import os
    
    # Detect Colab/Jupyter environment
    in_notebook = 'ipykernel' in sys.modules or 'google.colab' in sys.modules
    
    if in_notebook:
        # In Jupyter/Colab: reduce progress bar update frequency
        os.environ['TQDM_MININTERVAL'] = '10'  # Update every 10 seconds instead of every iteration
        os.environ['TQDM_MAXINTERVAL'] = '30'  # Max 30 seconds between updates
        print("[Colab] Progress bars configured for Jupyter/Colab (updates every 10s)")
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ds, train_df, valid_df, test_df = read_split(DATA_DIR)
    
    # Tính data statistics
    data_stats = get_data_stats(train_df, valid_df, test_df)
    print("\n[Data Statistics]")
    for split, stats in data_stats.items():
        print(f"  {split}: {stats['total']} samples "
              f"(valid: {stats['class_0']}/{stats['class_0_pct']:.1f}%, "
              f"invalid: {stats['class_1']}/{stats['class_1_pct']:.1f}%)")
    tokenizer = build_tokenizer()
    enc = encode_datasets(ds, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # ========== GPU OPTIMIZATIONS ==========
    # 1. Gradient Checkpointing: Trade compute for memory
    if USE_GRADIENT_CHECKPOINTING and torch.cuda.is_available():
        model.gradient_checkpointing_enable()
        print("[Optimization] Gradient checkpointing enabled")
    
    # 2. Torch Compile (PyTorch 2.x+): Faster forward/backward
    if USE_TORCH_COMPILE and torch.__version__ >= "2.0.0":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[Optimization] Torch compile enabled")
        except Exception as e:
            print(f"[Warning] Torch compile failed: {e}")
    
    # class weights nếu imbalance
    cw = get_class_weights(train_df)
    if cw is not None:
        print(f"[Info] Using class weights: class0={float(cw[0]):.3f}, class1={float(cw[1]):.3f}")

    fp16_flag = torch.cuda.is_available()
    bf16_flag = False
    tf32_flag = False
    
    # Check GPU architecture for optimizations
    if torch.cuda.is_available():
        # Get GPU capability
        gpu_capability = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[GPU] {gpu_name} (Compute Capability: {gpu_capability[0]}.{gpu_capability[1]})")
        
        # BF16: Ampere+ (compute capability >= 8.0)
        if gpu_capability[0] >= 8 and torch.cuda.is_bf16_supported():
            bf16_flag = True
            fp16_flag = False
            print("[Optimization] Using bfloat16 (Ampere+ GPU)")
        else:
            print("[Optimization] Using fp16 (pre-Ampere GPU)")
        
        # TF32: Only Ampere+ (compute capability >= 8.0)
        if gpu_capability[0] >= 8:
            tf32_flag = True
            print("[Optimization] TF32 enabled (Ampere+ GPU)")
        else:
            print("[Info] TF32 not available (requires Ampere or newer)")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_inv",  # ưu tiên F1 lớp invalid
        
        # Mixed precision training
        fp16=fp16_flag,
        bf16=bf16_flag,
        
        # Optimization settings
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",  # Fused optimizer (faster)
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        
        # Logging & saving (Colab-friendly!)
        logging_strategy="steps",
        logging_steps=100,  # ↑ Increased from 50 to reduce spam
        save_steps=0,
        disable_tqdm=False,  # Keep progress bars
        log_level="warning",  # ↓ Reduce INFO spam
        log_level_replica="warning",  # ↓ Reduce replica logs
        logging_first_step=False,  # Skip logging first step (reduce clutter)
        
        # DataLoader optimization (optimized for large datasets)
        dataloader_num_workers=0,  # ↓ Set to 0 for Colab (multiprocessing overhead on large data)
        dataloader_pin_memory=True,  # Faster CPU->GPU transfer
        dataloader_prefetch_factor=None,  # Auto-tune prefetch
        
        # Misc
        report_to="none",
        seed=SEED,
        
        # Speed optimizations (only for Ampere+ GPUs)
        tf32=tf32_flag,
    )

    # Trainer initialization (compatible with old and new transformers versions)
    trainer_kwargs = {
        "class_weight": cw,
        "model": model,
        "args": args,
        "train_dataset": enc["train"],
        "eval_dataset": enc["validation"],
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2), JsonLossLogger("out/metrics/train_log.ndjson")]
    }
    
    # Use 'processing_class' if available (transformers>=4.45), else 'tokenizer'
    import transformers
    from packaging import version
    if version.parse(transformers.__version__) >= version.parse("4.45.0"):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    
    trainer = WeightedTrainer(**trainer_kwargs)

    trainer.train()
    
    # Evaluate
    print("\n[Validation metrics (best ckpt)]:")
    best_val_metrics = trainer.evaluate(enc["validation"])
    print(best_val_metrics)

    print("\n[Test metrics]:")
    test_metrics = trainer.evaluate(enc["test"])
    print(test_metrics)

    # Save model
    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\n[Saved model] -> {SAVE_DIR}")
    
    # Generate comprehensive report
    print("\n[Generating Training Report...]")
    
    # Collect configuration
    config = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size_train": BATCH_TRAIN,
        "batch_size_eval": BATCH_EVAL,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_batch_size": BATCH_TRAIN * GRAD_ACCUM,
        "num_train_epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": 0.01,
        "seed": SEED,
        "fp16": fp16_flag,
        "bf16": bf16_flag,
        "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
        "optimizer": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        "tf32": tf32_flag,
    }
    
    # Load training history
    training_history = load_training_history("out/metrics/train_log.ndjson")
    
    # Class weights
    class_weights_list = cw.tolist() if cw is not None else None
    
    # Generate report
    generate_report(
        config=config,
        data_stats=data_stats,
        class_weights=class_weights_list,
        training_history=training_history,
        best_metrics=best_val_metrics,
        test_metrics=test_metrics,
        output_dir="out/metrics"
    )
    
    print("\n[Done] All training artifacts saved!")

if __name__ == "__main__":
    main()
