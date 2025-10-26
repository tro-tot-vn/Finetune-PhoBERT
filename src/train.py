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
BATCH_TRAIN = 8
BATCH_EVAL  = 16
GRAD_ACCUM  = 2          # giữ batch hiệu dụng lớn: BATCH_TRAIN * GRAD_ACCUM
EPOCHS      = 3
LR          = 2e-5

SEED        = 42
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

    def compute_loss(self, model, inputs, return_outputs=False):
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

    # class weights nếu imbalance
    cw = get_class_weights(train_df)
    if cw is not None:
        print(f"[Info] Using class weights: class0={float(cw[0]):.3f}, class1={float(cw[1]):.3f}")

    fp16_flag = torch.cuda.is_available()

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
        fp16=fp16_flag,
        logging_strategy="steps",
        logging_steps=50,
        save_steps=0,
        dataloader_num_workers=0,
        report_to="none",
        seed=SEED,
    )

    trainer = WeightedTrainer(
        class_weight=cw,
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), JsonLossLogger("out/metrics/train_log.ndjson")]
    )

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
