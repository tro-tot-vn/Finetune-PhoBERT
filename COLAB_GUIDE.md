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

### Cell 1.5 (OPTIONAL): Prepare VOZ-HSD Dataset
```python
# Skip this if you already have data in data/ folder
# This downloads and prepares balanced VOZ-HSD dataset

%cd /content/Finetune-PhoBERT/data

# Option A: Default (70-30, probs≥0.95) - Recommended
!python prepare_voz_balanced.py

# Option B: More conservative (80-20, better UX)
# !python prepare_voz_balanced.py --confidence 0.95 --ratio 80

# Option C: Higher recall (60-40, catch more hate)
# !python prepare_voz_balanced.py --confidence 0.90 --ratio 60

# Check summary
!cat dataset_summary.txt

%cd /content/Finetune-PhoBERT
```

**Expected time:** 5-10 minutes for download + processing  
**Output:** train.csv, valid.csv, test.csv (~547K total samples)

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

## ⚠️ Common Issues & Warnings

### Normal Warnings (OK to ignore)
```
E0000 00:00:xxx ... Unable to register cuFFT factory...
E0000 00:00:xxx ... Unable to register cuDNN factory...
FutureWarning: `tokenizer` is deprecated...
```
**These are normal** - TensorFlow backend warnings và deprecation warnings. Code vẫn chạy bình thường!

### Issue 1: `TypeError: evaluation_strategy` 
**Fixed!** Code đã update cho transformers mới.

### Issue 2: `TypeError: compute_loss() got unexpected keyword` 
**Fixed!** Code đã tương thích với transformers>=4.35.0.

### Issue 3: `FileNotFoundError: data/`
**Solution**: Chắc chắn đã chạy `%cd /content/Finetune-PhoBERT` trước!

### Issue 4: CUDA out of memory
**Solutions**:
- Giảm `BATCH_TRAIN` trong `src/train.py` (line 27)
- Giảm `MAX_LENGTH` (line 26)
- Dùng Runtime > Change runtime type > T4 GPU

### Issue 5: Session timeout
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

## ⚡ GPU Optimization (T4) - AGGRESSIVE! 🔥

**Code đã được optimize MẠNH để ăn đầy VRAM!**
- ✅ Batch size MAX (32/64) - no more 2.7GB waste!
- ✅ FP16 mixed precision
- ✅ Gradient checkpointing OFF (for speed)
- ✅ Fused optimizer
- ✅ Parallel data loading

**Expected:**
- VRAM usage: **~12-13GB / 16GB** (80%+) ✅
- Training speed: **~8-10 phút** cho 24K samples (3 epochs) 🚀
- GPU utilization: **85-95%**

### Monitor GPU (IMPORTANT):
```python
!watch -n 1 nvidia-smi  # Live monitoring (Ctrl+C to stop)
```

**Should see:**
- Memory: **12-13GB / 16GB** (not 2.7GB!)
- GPU Util: **85-95%**

### If OOM (out of memory):
Edit `src/train.py` lines 28-29:
```python
BATCH_TRAIN = 24  # Giảm từ 32
BATCH_EVAL = 48   # Giảm từ 64
```

### If VRAM < 10GB (still wasting):
```python
BATCH_TRAIN = 40  # Tăng lên!
BATCH_EVAL = 80
```

Chi tiết: Xem [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md)

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

