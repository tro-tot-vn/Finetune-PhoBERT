"""
Training Curves Visualizer
Visualize loss và metrics evolution trong quá trình training.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np


def load_training_log(ndjson_path: str):
    """Đọc và parse training log từ NDJSON."""
    if not os.path.exists(ndjson_path):
        raise FileNotFoundError(f"Training log not found: {ndjson_path}")
    
    logs = []
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    
    return logs


def extract_metrics(logs):
    """Tách metrics từ logs thành các arrays."""
    # Separate training steps and evaluation steps
    train_steps = []
    train_loss = []
    
    eval_epochs = []
    eval_steps = []
    eval_loss = []
    eval_accuracy = []
    eval_precision = []
    eval_recall = []
    eval_f1 = []
    
    for entry in logs:
        step = entry.get("step", 0)
        epoch = entry.get("epoch")
        
        # Training loss (logged at intervals)
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        
        # Evaluation metrics (logged per epoch)
        if "eval_loss" in entry:
            eval_steps.append(step)
            if epoch is not None:
                eval_epochs.append(epoch)
            eval_loss.append(entry["eval_loss"])
            
            if "eval_accuracy" in entry:
                eval_accuracy.append(entry["eval_accuracy"])
            if "eval_precision_inv" in entry:
                eval_precision.append(entry["eval_precision_inv"])
            if "eval_recall_inv" in entry:
                eval_recall.append(entry["eval_recall_inv"])
            if "eval_f1_inv" in entry:
                eval_f1.append(entry["eval_f1_inv"])
    
    return {
        "train": {"steps": train_steps, "loss": train_loss},
        "eval": {
            "steps": eval_steps,
            "epochs": eval_epochs,
            "loss": eval_loss,
            "accuracy": eval_accuracy,
            "precision": eval_precision,
            "recall": eval_recall,
            "f1": eval_f1,
        }
    }


def plot_training_curves(metrics, output_path="out/metrics/training_curves.png"):
    """
    Tạo comprehensive training curves plot.
    
    Args:
        metrics: Dict chứa train và eval metrics
        output_path: Đường dẫn lưu plot
    """
    train_data = metrics["train"]
    eval_data = metrics["eval"]
    
    # Tạo figure với 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # ========== Plot 1: Loss Curves ==========
    ax1 = axes[0]
    
    # Training loss (smooth line)
    if train_data["loss"]:
        ax1.plot(train_data["steps"], train_data["loss"], 
                label="Train Loss", color="#3498db", linewidth=1.5, alpha=0.7)
    
    # Validation loss (with markers)
    if eval_data["loss"]:
        ax1.plot(eval_data["steps"], eval_data["loss"], 
                label="Validation Loss", color="#e74c3c", 
                linewidth=2, marker="o", markersize=6)
    
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--")
    
    # Annotate best validation loss
    if eval_data["loss"]:
        best_val_loss = min(eval_data["loss"])
        best_idx = eval_data["loss"].index(best_val_loss)
        best_step = eval_data["steps"][best_idx]
        ax1.plot(best_step, best_val_loss, "r*", markersize=15, 
                label=f"Best: {best_val_loss:.4f}")
        ax1.legend(loc="upper right", fontsize=10)
    
    # ========== Plot 2: Metrics Evolution ==========
    ax2 = axes[1]
    
    if eval_data["epochs"]:
        x_vals = eval_data["epochs"]
        x_label = "Epoch"
    else:
        x_vals = eval_data["steps"]
        x_label = "Step"
    
    if eval_data["accuracy"]:
        ax2.plot(x_vals, eval_data["accuracy"], 
                label="Accuracy", color="#2ecc71", linewidth=2, marker="s", markersize=5)
    
    if eval_data["precision"]:
        ax2.plot(x_vals, eval_data["precision"], 
                label="Precision (invalid)", color="#9b59b6", linewidth=2, marker="^", markersize=5)
    
    if eval_data["recall"]:
        ax2.plot(x_vals, eval_data["recall"], 
                label="Recall (invalid)", color="#e67e22", linewidth=2, marker="v", markersize=5)
    
    if eval_data["f1"]:
        ax2.plot(x_vals, eval_data["f1"], 
                label="F1 Score (invalid)", color="#e74c3c", linewidth=2.5, marker="o", markersize=6)
    
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Validation Metrics Evolution", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_ylim([0, 1.05])
    
    # Annotate best F1
    if eval_data["f1"]:
        best_f1 = max(eval_data["f1"])
        best_idx = eval_data["f1"].index(best_f1)
        best_x = x_vals[best_idx]
        ax2.plot(best_x, best_f1, "r*", markersize=15)
        ax2.annotate(f"Best F1: {best_f1:.4f}", 
                    xy=(best_x, best_f1), 
                    xytext=(10, -20), textcoords="offset points",
                    fontsize=9, color="red", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
                    arrowprops=dict(arrowstyle="->", color="red"))
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[Saved] Training curves: {output_path}")


def main():
    """Main entry point."""
    log_path = "out/metrics/train_log.ndjson"
    output_path = "out/metrics/training_curves.png"
    
    if not os.path.exists(log_path):
        print(f"[Error] Training log not found: {log_path}")
        print("Please train the model first to generate training logs.")
        return
    
    print(f"[Loading] {log_path}")
    logs = load_training_log(log_path)
    print(f"[Loaded] {len(logs)} log entries")
    
    metrics = extract_metrics(logs)
    print(f"[Extracted] {len(metrics['train']['loss'])} training steps, "
          f"{len(metrics['eval']['loss'])} evaluation points")
    
    plot_training_curves(metrics, output_path)
    print("[Done]")


if __name__ == "__main__":
    main()

