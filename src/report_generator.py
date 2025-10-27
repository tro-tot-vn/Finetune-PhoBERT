"""
Training Report Generator
Tạo báo cáo chi tiết về quá trình training cho research paper/report.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


def load_training_history(ndjson_path: str) -> Dict[str, list]:
    """Đọc training log từ NDJSON file."""
    if not os.path.exists(ndjson_path):
        return {}
    
    history = {
        "steps": [],
        "epochs": [],
        "train_loss": [],
        "eval_loss": [],
        "eval_accuracy": [],
        "eval_precision_inv": [],
        "eval_recall_inv": [],
        "eval_f1_inv": [],
    }
    
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            
            if "step" in entry:
                history["steps"].append(entry["step"])
            if "epoch" in entry and entry["epoch"] is not None:
                history["epochs"].append(entry["epoch"])
            if "loss" in entry:
                history["train_loss"].append(entry["loss"])
            if "eval_loss" in entry:
                history["eval_loss"].append(entry["eval_loss"])
            if "eval_accuracy" in entry:
                history["eval_accuracy"].append(entry["eval_accuracy"])
            if "eval_precision_inv" in entry:
                history["eval_precision_inv"].append(entry["eval_precision_inv"])
            if "eval_recall_inv" in entry:
                history["eval_recall_inv"].append(entry["eval_recall_inv"])
            if "eval_f1_inv" in entry:
                history["eval_f1_inv"].append(entry["eval_f1_inv"])
    
    return history


def generate_report(
    config: Dict[str, Any],
    data_stats: Dict[str, Any],
    class_weights: Optional[list],
    training_history: Dict[str, list],
    best_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    output_dir: str = "out/metrics"
) -> None:
    """
    Tạo báo cáo training đầy đủ.
    
    Args:
        config: Hyperparameters (learning_rate, batch_size, epochs, etc.)
        data_stats: Thống kê data (train/val/test sizes, class distribution)
        class_weights: Class weights nếu có
        training_history: Metrics theo epochs
        best_metrics: Best validation metrics
        test_metrics: Final test set metrics
        output_dir: Thư mục lưu báo cáo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo structured report (JSON)
    report = {
        "generated_at": datetime.now().isoformat(),
        "model_config": config,
        "data_statistics": data_stats,
        "class_weights": class_weights,
        "training_history": training_history,
        "best_validation_metrics": best_metrics,
        "test_metrics": test_metrics,
    }
    
    json_path = os.path.join(output_dir, "training_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Tạo human-readable markdown report
    md_path = os.path.join(output_dir, "training_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Training Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model Configuration
        f.write("## Model Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        for key, val in config.items():
            f.write(f"| {key} | {val} |\n")
        f.write("\n")
        
        # Data Statistics
        f.write("## Data Statistics\n\n")
        f.write("| Split | Size | Class 0 (valid) | Class 1 (invalid) |\n")
        f.write("|-------|------|-----------------|-------------------|\n")
        for split in ["train", "validation", "test"]:
            if split in data_stats:
                stats = data_stats[split]
                f.write(f"| {split.capitalize()} | {stats['total']} | "
                       f"{stats.get('class_0', 'N/A')} ({stats.get('class_0_pct', 0):.1f}%) | "
                       f"{stats.get('class_1', 'N/A')} ({stats.get('class_1_pct', 0):.1f}%) |\n")
        f.write("\n")
        
        # Class Weights
        if class_weights:
            f.write("## Class Weights\n\n")
            f.write("Applied weighted loss for imbalanced data:\n\n")
            f.write(f"- **Class 0 (valid):** {class_weights[0]:.4f}\n")
            f.write(f"- **Class 1 (invalid):** {class_weights[1]:.4f}\n\n")
        
        # Training Summary
        f.write("## Training Summary\n\n")
        if training_history.get("train_loss"):
            f.write(f"- **Total epochs:** {config.get('num_train_epochs', 'N/A')}\n")
            f.write(f"- **Final train loss:** {training_history['train_loss'][-1]:.4f}\n")
            if training_history.get("eval_loss"):
                f.write(f"- **Final validation loss:** {training_history['eval_loss'][-1]:.4f}\n")
        f.write("\n")
        
        # Best Validation Metrics
        f.write("## Best Validation Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, val in best_metrics.items():
            f.write(f"| {key} | {val:.4f} |\n")
        f.write("\n")
        
        # Test Set Results
        f.write("## Test Set Results\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for key, val in test_metrics.items():
            f.write(f"| {key} | {val:.4f} |\n")
        f.write("\n")
        
        # Files Reference
        f.write("## Generated Files\n\n")
        f.write("Visualization files:\n\n")
        f.write("- `training_curves.png` - Loss and metrics evolution\n")
        f.write("- `confusion_matrix.png` - Confusion matrix (counts)\n")
        f.write("- `confusion_matrix_norm.png` - Confusion matrix (normalized)\n")
        f.write("- `roc_curve.png` - ROC curve with AUC\n")
        f.write("- `pr_curve.png` - Precision-Recall curve\n")
        f.write("- `threshold_analysis.png` - Metrics at different thresholds\n")
        f.write("- `prediction_dist.png` - Prediction confidence distribution\n")
        f.write("\n")
    
    print(f"[Report Generated]")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    # Example usage
    history = load_training_history("out/metrics/train_log.ndjson")
    if history:
        print("Loaded training history:")
        for key, values in history.items():
            if values:
                print(f"  {key}: {len(values)} entries")

