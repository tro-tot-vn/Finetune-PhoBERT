# Google Colab Guide

## 🚀 Quick Start

### Cell 1: Setup
```python
# Clone repository
!git clone https://github.com/your-username/Finetune-PhoBERT.git

# QUAN TRỌNG: CD vào project (dùng % không phải !)
%cd /content/Finetune-PhoBERT

# Install dependencies
!pip install -q -r requirements.txt
```

### Cell 2: Check Data
```python
import pandas as pd

train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train: {len(train_df)} samples")
print(f"Valid: {len(valid_df)} samples")
print(f"Test: {len(test_df)} samples")
print(f"\nLabel distribution:\n{train_df['label'].value_counts()}")
```

### Cell 3: Train Model
```python
# Train (takes ~15-30 mins with GPU)
!python src/train.py
```

### Cell 4: Visualize Training
```python
# Generate training curves
!python src/plot_training.py

# Display
from IPython.display import Image, display
display(Image('out/metrics/training_curves.png'))
```

### Cell 5: Evaluate
```python
# Run evaluation (generates all plots)
!python src/evaluation.py
```

### Cell 6: View Results
```python
from IPython.display import Image, display

# Confusion Matrix
print("Confusion Matrix (Normalized):")
display(Image('out/metrics/confusion_matrix_norm.png'))

# ROC Curve
print("\nROC Curve:")
display(Image('out/metrics/roc_curve.png'))

# PR Curve (important for imbalanced data!)
print("\nPrecision-Recall Curve:")
display(Image('out/metrics/pr_curve.png'))

# Threshold Analysis
print("\nThreshold Analysis:")
display(Image('out/metrics/threshold_analysis.png'))

# Prediction Distribution
print("\nPrediction Distribution:")
display(Image('out/metrics/prediction_dist.png'))
```

### Cell 7: View Report
```python
# Training report
!cat out/metrics/training_report.md

# Metrics JSON
import json
with open('out/metrics/report.json', 'r') as f:
    print(json.dumps(json.load(f), indent=2))
```

### Cell 8: Download Results
```python
# Zip all results
!zip -r results.zip out/metrics/ models/phobert-moderation/

# Download
from google.colab import files
files.download('results.zip')
```

---

## ⚠️ Common Issues

### Issue 1: `TypeError: evaluation_strategy` 
**Fixed!** Code đã update cho transformers mới.

### Issue 2: `FileNotFoundError: data/`
**Solution**: Chắc chắn đã chạy `%cd /content/Finetune-PhoBERT` trước!

### Issue 3: CUDA out of memory
**Solutions**:
- Giảm `BATCH_TRAIN` trong `src/train.py` (line 27)
- Giảm `MAX_LENGTH` (line 26)
- Dùng Runtime > Change runtime type > T4 GPU

### Issue 4: Session timeout
**Solution**: 
- Download model thường xuyên
- Hoặc mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
# Sau đó copy kết quả vào Drive
!cp -r out/ /content/drive/MyDrive/phobert-results/
```

---

## 🎯 Tips

1. **Enable GPU**: Runtime > Change runtime type > GPU (T4)
2. **Monitor training**: Training logs hiển thị realtime
3. **Save checkpoints**: Model tự động lưu best checkpoint
4. **Download early**: Download results ngay sau train để tránh mất data

---

## 📊 Expected Output Structure

```
/content/Finetune-PhoBERT/
├── out/metrics/
│   ├── training_report.json
│   ├── training_report.md
│   ├── train_log.ndjson
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_norm.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── threshold_analysis.png
│   ├── prediction_dist.png
│   └── report.json
└── models/phobert-moderation/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    └── ...
```

All files ready for your research report! 🎉

