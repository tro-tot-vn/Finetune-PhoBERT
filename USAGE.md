# Hướng dẫn sử dụng - Training Report System

## 📋 Tổng quan

Hệ thống đã được nâng cấp với các công cụ visualization và reporting toàn diện cho báo cáo nghiên cứu.

## 🚀 Cài đặt

```bash
# Cài đặt dependencies (đã thêm seaborn)
pip install -r requirements.txt
```

## 📊 Workflow

### 1. Training Model

```bash
cd src
python train.py
```

**Output sau khi train:**
- Model được lưu tại: `models/phobert-moderation/`
- Training logs: `out/metrics/train_log.ndjson`
- **Training report**: `out/metrics/training_report.json` và `training_report.md`

### 2. Visualize Training Curves

```bash
cd src
python plot_training.py
```

**Output:**
- `out/metrics/training_curves.png` - Loss curves & metrics evolution

### 3. Evaluate trên Test Set

```bash
cd src
python evaluation.py
```

**Output:**
- `out/metrics/confusion_matrix.png` - Confusion matrix (counts)
- `out/metrics/confusion_matrix_norm.png` - Confusion matrix (normalized %)
- `out/metrics/roc_curve.png` - ROC curve với AUC
- `out/metrics/pr_curve.png` - **Precision-Recall curve** (quan trọng!)
- `out/metrics/threshold_analysis.png` - Metrics @ different thresholds
- `out/metrics/prediction_dist.png` - Prediction confidence distribution
- `out/metrics/report.json` - Test metrics JSON

## 📁 Cấu trúc Output

```
out/metrics/
├── training_report.json       # Training config & metrics (JSON)
├── training_report.md          # Human-readable report (Markdown)
├── train_log.ndjson           # Raw training logs
├── training_curves.png         # Training visualization
├── confusion_matrix.png        # CM counts
├── confusion_matrix_norm.png   # CM percentages
├── roc_curve.png              # ROC with threshold marker
├── pr_curve.png               # PR curve (for imbalanced data)
├── threshold_analysis.png      # Find optimal threshold
├── prediction_dist.png         # Confidence distribution
└── report.json                # Test set metrics
```

## 🎯 Key Features

### 1. **Training Report** (`training_report.md`)

Bao gồm:
- ✅ Hyperparameters (learning rate, batch size, epochs, etc.)
- ✅ Data statistics (class distribution cho train/val/test)
- ✅ Class weights (nếu có)
- ✅ Training summary (loss curves)
- ✅ Best validation metrics
- ✅ Test set results

### 2. **Enhanced Visualizations**

#### Confusion Matrix
- 2 versions: counts + normalized percentages
- Color-coded với seaborn palette
- Annotations rõ ràng

#### ROC Curve
- Grid để dễ đọc
- Mark threshold hiện tại
- AUC score

#### **PR Curve (MỚI)**
- **Critical cho imbalanced data!**
- Shows precision-recall tradeoff
- Baseline comparison

#### **Threshold Analysis (MỚI)**
- Plot metrics @ 101 thresholds (0.0 to 1.0)
- Tự động tìm best F1 threshold
- So sánh với default 0.5

#### **Prediction Distribution (MỚI)**
- Histogram of prediction scores
- Separated by true class
- Helps understand model confidence

### 3. **Training Curves**

- Loss evolution (train + validation)
- Metrics evolution (accuracy, precision, recall, F1)
- Mark best checkpoint
- Annotations rõ ràng

## 🔧 Customization

### Thay đổi threshold (evaluation)

```bash
THRESHOLD=0.6 python evaluation.py
```

### Modify configs (training)

Edit `src/train.py`:
```python
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 192
BATCH_TRAIN = 8
EPOCHS = 3
LR = 2e-5
```

## 📝 Cho báo cáo nghiên cứu

### Files cần thiết:

1. **Training Report**: `out/metrics/training_report.md` 
   - Copy trực tiếp vào báo cáo hoặc paper appendix

2. **Visualizations**:
   - `training_curves.png` - Quá trình training
   - `confusion_matrix_norm.png` - Confusion matrix %
   - `pr_curve.png` - Performance cho imbalanced data
   - `threshold_analysis.png` - Threshold tuning

3. **Metrics JSON**: `training_report.json` + `report.json`
   - Parse để tạo LaTeX tables

### LaTeX Table Example:

```latex
\begin{table}[h]
\centering
\caption{Test Set Performance}
\begin{tabular}{lc}
\hline
Metric & Score \\
\hline
Accuracy & 0.XXX \\
Precision (invalid) & 0.XXX \\
Recall (invalid) & 0.XXX \\
F1 Score (invalid) & 0.XXX \\
ROC AUC & 0.XXX \\
PR AUC & 0.XXX \\
\hline
\end{tabular}
\end{table}
```

## ⚠️ Notes

- Seaborn warning khi chưa install: bình thường, chạy `pip install seaborn`
- Training curves cần có `train_log.ndjson` từ quá trình training
- Evaluation cần có trained model tại `models/phobert-moderation/`

## 🎉 Done!

Bây giờ bạn có đầy đủ metrics và visualizations cho báo cáo!

