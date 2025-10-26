import os, json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt

# ========== CẤU HÌNH ==========
MODEL_DIR = "models/phobert-moderation"
TEST_CSV  = "data/test.csv"
MAX_LENGTH = 192
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))  # có thể set THRESHOLD
OUT_DIR = "out/metrics"
os.makedirs(OUT_DIR, exist_ok=True)
# ==============================

def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    torch.set_grad_enabled(False)
    return tok, model

def predict_probs(tokenizer, model, texts):
    probs = []
    for s in texts:
        enc = tokenizer(s.strip(), return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1)[0, 1].item()  # prob lớp 1 (invalid)
        probs.append(p)
    return np.array(probs)

def plot_confusion_matrix(cm, labels, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    # chú thích từng ô
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve (positive = invalid)")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return roc_auc

def main():
    # 1) Load data & model
    df = pd.read_csv(TEST_CSV)
    df["text"] = df["text"].astype(str)
    y_true = df["label"].astype(int).values

    tokenizer, model = load_model()

    # 2) Dự đoán xác suất cho lớp 1 (invalid)
    y_score = predict_probs(tokenizer, model, df["text"].tolist())

    # 3) Nhị phân theo THRESHOLD
    y_pred = (y_score >= THRESHOLD).astype(int)

    # 4) Tính chỉ số
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 5) Vẽ & lưu
    cm_path  = os.path.join(OUT_DIR, "confusion_matrix.png")
    roc_path = os.path.join(OUT_DIR, "roc_curve.png")
    plot_confusion_matrix(cm, labels=["valid(0)", "invalid(1)"], path=cm_path)
    roc_auc = plot_roc(y_true, y_score, path=roc_path)

    # 6) Lưu báo cáo JSON
    report = {
        "threshold": THRESHOLD,
        "accuracy": acc,
        "precision_inv": prec,
        "recall_inv": rec,
        "f1_inv": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels_order": ["actual_0_valid", "actual_1_invalid"],
            "matrix": cm.tolist()
        }
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[Saved]", cm_path)
    print("[Saved]", roc_path)
    print("[Saved]", os.path.join(OUT_DIR, "report.json"))

if __name__ == "__main__":
    main()
