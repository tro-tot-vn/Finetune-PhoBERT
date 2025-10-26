import os, json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_confusion_matrix(cm, labels, path, normalize=False):
    """
    Plot confusion matrix với colormap đẹp hơn.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        path: Output path
        normalize: If True, normalize to percentages
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        fmt = ".1f"
        cbar_label = "Percentage (%)"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"
    
    # Use seaborn's color palette
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    
    # Tick marks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    ax.set_ylabel("Actual", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
    
    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    
    # Annotate cells
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                text = f"{cm_display[i, j]:.1f}%\n({cm[i, j]})"
            else:
                text = format(cm[i, j], "d")
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black",
                    fontsize=11, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_roc(y_true, y_score, path, threshold=0.5):
    """
    Plot ROC curve với grid và threshold marker.
    
    Args:
        y_true: True labels
        y_score: Prediction scores
        path: Output path
        threshold: Current threshold to mark on curve
    
    Returns:
        roc_auc: AUC score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # ROC curve
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", 
            color="#3498db", linewidth=2.5)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", 
            linewidth=1.5, label="Random Classifier")
    
    # Mark current threshold
    idx = np.argmin(np.abs(thresholds - threshold))
    ax.plot(fpr[idx], tpr[idx], "r*", markersize=15, 
            label=f"Threshold = {threshold:.2f}")
    
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curve (positive class = invalid)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return roc_auc

def plot_pr_curve(y_true, y_score, path):
    """
    Plot Precision-Recall curve - quan trọng cho imbalanced data!
    
    Args:
        y_true: True labels
        y_score: Prediction scores
        path: Output path
    
    Returns:
        pr_auc: Average Precision score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=1)
    pr_auc = auc(recall, precision)
    
    # Tính baseline (tỉ lệ class positive)
    baseline = np.sum(y_true) / len(y_true)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # PR curve
    ax.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.3f})", 
            color="#e74c3c", linewidth=2.5)
    
    # Baseline (no skill classifier)
    ax.axhline(y=baseline, linestyle="--", color="gray", 
               linewidth=1.5, label=f"Baseline = {baseline:.3f}")
    
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curve (positive class = invalid)", 
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pr_auc

def plot_threshold_analysis(y_true, y_score, path):
    """
    Analyze metrics ở nhiều threshold khác nhau.
    Giúp tìm optimal threshold cho use case cụ thể.
    
    Args:
        y_true: True labels
        y_score: Prediction scores
        path: Output path
    """
    thresholds = np.linspace(0, 1, 101)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, pos_label=1, average="binary", zero_division=0
        )
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(thresholds, accuracies, label="Accuracy", 
            color="#2ecc71", linewidth=2, linestyle="-")
    ax.plot(thresholds, precisions, label="Precision (invalid)", 
            color="#9b59b6", linewidth=2, linestyle="--")
    ax.plot(thresholds, recalls, label="Recall (invalid)", 
            color="#e67e22", linewidth=2, linestyle="-.")
    ax.plot(thresholds, f1_scores, label="F1 Score (invalid)", 
            color="#e74c3c", linewidth=2.5, linestyle="-")
    
    # Mark best F1 threshold
    best_f1_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    ax.axvline(x=best_thresh, linestyle=":", color="red", linewidth=2, alpha=0.7)
    ax.plot(best_thresh, best_f1, "r*", markersize=15)
    ax.annotate(f"Best F1 @ {best_thresh:.2f}\nF1 = {best_f1:.3f}", 
                xy=(best_thresh, best_f1), 
                xytext=(20, -20), textcoords="offset points",
                fontsize=9, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.6),
                arrowprops=dict(arrowstyle="->", color="red"))
    
    # Mark default 0.5 threshold
    ax.axvline(x=0.5, linestyle=":", color="gray", linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel("Threshold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Metrics vs Threshold", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"[Analysis] Best F1 threshold: {best_thresh:.3f} (F1 = {best_f1:.4f})")

def plot_prediction_distribution(y_true, y_score, path):
    """
    Visualize phân bố prediction scores cho từng class.
    Giúp hiểu confidence của model.
    
    Args:
        y_true: True labels
        y_score: Prediction scores
        path: Output path
    """
    # Separate scores by true class
    scores_valid = y_score[y_true == 0]  # true valid
    scores_invalid = y_score[y_true == 1]  # true invalid
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Histograms
    ax.hist(scores_valid, bins=50, alpha=0.6, color="#2ecc71", 
            label=f"True Valid (n={len(scores_valid)})", edgecolor="black")
    ax.hist(scores_invalid, bins=50, alpha=0.6, color="#e74c3c", 
            label=f"True Invalid (n={len(scores_invalid)})", edgecolor="black")
    
    # Mark threshold
    ax.axvline(x=0.5, linestyle="--", color="black", linewidth=2, 
               label="Default Threshold (0.5)")
    
    ax.set_xlabel("Predicted Probability (Invalid Class)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax.set_title("Prediction Score Distribution by True Class", 
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")
    
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

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

    # 5) Vẽ & lưu visualizations
    print("\n[Creating Visualizations...]")
    
    # Confusion matrices (both versions)
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    cm_norm_path = os.path.join(OUT_DIR, "confusion_matrix_norm.png")
    plot_confusion_matrix(cm, labels=["valid(0)", "invalid(1)"], path=cm_path, normalize=False)
    plot_confusion_matrix(cm, labels=["valid(0)", "invalid(1)"], path=cm_norm_path, normalize=True)
    
    # ROC curve
    roc_path = os.path.join(OUT_DIR, "roc_curve.png")
    roc_auc = plot_roc(y_true, y_score, path=roc_path, threshold=THRESHOLD)
    
    # PR curve (critical for imbalanced data!)
    pr_path = os.path.join(OUT_DIR, "pr_curve.png")
    pr_auc = plot_pr_curve(y_true, y_score, path=pr_path)
    
    # Threshold analysis
    thresh_path = os.path.join(OUT_DIR, "threshold_analysis.png")
    plot_threshold_analysis(y_true, y_score, path=thresh_path)
    
    # Prediction distribution
    dist_path = os.path.join(OUT_DIR, "prediction_dist.png")
    plot_prediction_distribution(y_true, y_score, path=dist_path)

    # 6) Lưu báo cáo JSON
    report = {
        "threshold": THRESHOLD,
        "accuracy": acc,
        "precision_inv": prec,
        "recall_inv": rec,
        "f1_inv": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "labels_order": ["actual_0_valid", "actual_1_invalid"],
            "matrix": cm.tolist()
        }
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[Saved Visualizations]")
    print(f"  {cm_path}")
    print(f"  {cm_norm_path}")
    print(f"  {roc_path}")
    print(f"  {pr_path}")
    print(f"  {thresh_path}")
    print(f"  {dist_path}")
    print(f"\n[Saved Report]")
    print(f"  {os.path.join(OUT_DIR, 'report.json')}")

if __name__ == "__main__":
    main()
