# Finetune-PhoBERT for Vietnamese Hate Speech Detection

Fine-tuning PhoBERT-base-v2 cho Vietnamese text moderation (binary classification: clean/hate speech) với VOZ-HSD dataset.

## 🎯 Use Case

**Pre-screening system cho content moderation:**
```
User posts → PhoBERT (auto-filter) → Human review → Publish/Reject
```

Optimized for **high precision** (minimize false positives = better UX)

## ⭐ Key Features

### 🚀 Performance
- ✅ **3-4x faster training** on T4 GPU (optimized batch sizes & mixed precision)
- ✅ **10-20x faster inference** (batch prediction instead of single samples)
- ✅ **Expected F1: 75-80%** on balanced VOZ-HSD dataset

### 📊 Data Processing
- ✅ **VOZ-HSD integration** (10.7M samples with confidence scores)
- ✅ **Confidence-based filtering** (probs ≥ 0.95 for quality)
- ✅ **Balanced sampling** (70-30 clean-hate for better learning)
- ✅ **Stratified splits** (80-10-10 train-val-test)

### 🎨 Visualization & Reporting
- ✅ **8 comprehensive plots** (confusion matrix, ROC, PR curve, threshold analysis, etc.)
- ✅ **Training curves** with metrics evolution
- ✅ **Automated reports** (JSON + Markdown)
- ✅ **Threshold analysis** for production optimization

### ⚡ GPU Optimization
- ✅ **Auto-detect GPU architecture** (T4, V100, A100, etc.)
- ✅ **Smart mixed precision** (FP16 for Turing, BF16 for Ampere)
- ✅ **Aggressive batch sizes** (32/64 for max VRAM usage)
- ✅ **No gradient checkpointing** (trade memory for speed on T4)

## 🚀 Quick Start

### Option A: Use Existing Data
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (if you already have data/train.csv, valid.csv, test.csv)
python src/train.py

# 3. Visualize & evaluate
python src/plot_training.py
python src/evaluation.py
```

### Option B: Prepare VOZ-HSD Dataset (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare balanced dataset from VOZ-HSD
cd data
python prepare_voz_balanced.py --confidence 0.95 --ratio 70

# 3. Train model
cd ..
python src/train.py

# 4. Visualize & evaluate
python src/plot_training.py
python src/evaluation.py
```

**Expected time:**
- Data preparation: 5-10 minutes (first time only)
- Training (547K samples): ~1.5-2 hours on T4 GPU
- Evaluation: ~30 seconds

## 📊 Expected Performance

| Dataset Size | Training Time (T4) | F1 Score | Precision | Recall |
|--------------|-------------------|----------|-----------|---------|
| 33K (small) | 6 min | 70% | 65% | 77% |
| **547K (default)** | **1.5 hours** | **75-80%** | **75-80%** | **75-80%** |
| 820K (conservative) | 2 hours | 77-82% | 78-83% | 75-80% |

## 📁 Project Structure

```
.
├── data/
│   ├── prepare_voz_balanced.py    # VOZ-HSD data processor ⭐
│   ├── normalize_labels_flex.py   # Label normalization utility
│   ├── README_VOZ.md              # VOZ-HSD preparation guide
│   ├── train.csv                  # Training set (generated)
│   ├── valid.csv                  # Validation set (generated)
│   └── test.csv                   # Test set (generated)
│
├── src/
│   ├── train.py                   # Training script (GPU optimized)
│   ├── evaluation.py              # Evaluation + 8 visualizations
│   ├── plot_training.py           # Training curves plotter
│   ├── report_generator.py        # Auto report generation
│   └── callbacks.py               # Custom training callbacks
│
├── out/                           # Generated outputs
│   └── metrics/
│       ├── training_report.md     # Human-readable report
│       ├── training_report.json   # Structured metrics
│       ├── training_curves.png    # Loss & metrics evolution
│       ├── confusion_matrix.png   # CM (counts)
│       ├── confusion_matrix_norm.png  # CM (normalized)
│       ├── roc_curve.png          # ROC with AUC
│       ├── pr_curve.png           # Precision-Recall curve ⭐
│       ├── threshold_analysis.png # Optimal threshold finder
│       └── prediction_dist.png    # Confidence distribution
│
├── models/
│   └── phobert-moderation/        # Saved model
│
├── USAGE.md                       # Detailed usage guide
├── COLAB_GUIDE.md                 # Google Colab instructions
├── GPU_OPTIMIZATION.md            # GPU tuning guide
└── requirements.txt
```

## 🔧 Configuration

### Data Preparation (`data/prepare_voz_balanced.py`):
```bash
# Default: 70-30 balance, probs≥0.95
python prepare_voz_balanced.py

# Conservative (better UX): 80-20 balance
python prepare_voz_balanced.py --ratio 80

# Higher recall: 60-40 balance
python prepare_voz_balanced.py --ratio 60
```

### Training (`src/train.py`):
```python
# Key hyperparameters (optimized for T4):
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 192
BATCH_TRAIN = 32  # Aggressive for T4!
BATCH_EVAL = 64
EPOCHS = 3
LR = 2e-5
USE_GRADIENT_CHECKPOINTING = False  # Speed over memory
```

### Evaluation:
- Batch prediction (64 samples at a time)
- Multiple thresholds analysis (0.0 to 1.0)
- 8 comprehensive visualizations

## 📚 Documentation

- **[USAGE.md](USAGE.md)** - Detailed usage instructions
- **[COLAB_GUIDE.md](COLAB_GUIDE.md)** - Google Colab step-by-step guide
- **[GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md)** - GPU tuning & troubleshooting
- **[data/README_VOZ.md](data/README_VOZ.md)** - VOZ-HSD dataset preparation

## 🎓 For Research Papers

### Citing this work:
If using VOZ-HSD dataset, cite:
```bibtex
@inproceedings{thanh-nguyen-2024-vihatet5,
    title = "{V}i{H}ate{T}5: Enhancing Hate Speech Detection in {V}ietnamese With a Unified Text-to-Text Transformer Model",
    author = "Thanh Nguyen, Luan",
    booktitle = "Findings of ACL 2024",
    year = "2024",
    url = "https://aclanthology.org/2024.findings-acl.355"
}
```

### Key metrics for papers:
- Dataset: VOZ-HSD (confidence ≥ 0.95, balanced 70-30)
- Model: PhoBERT-base-v2 (135M parameters)
- Training: 547K samples, 3 epochs, ~1.5 hours on T4
- Performance: F1 75-80%, Precision 75-80%, Recall 75-80%

### Generated outputs for papers:
- `training_report.md` - Copy to appendix
- `confusion_matrix_norm.png` - For results section
- `pr_curve.png` - For performance analysis
- `threshold_analysis.png` - For deployment discussion

## 🚀 Google Colab

See [COLAB_GUIDE.md](COLAB_GUIDE.md) for full instructions.

**Quick start:**
```python
# In Colab
!git clone https://github.com/your-repo/Finetune-PhoBERT.git
%cd Finetune-PhoBERT
!pip install -q -r requirements.txt

# Prepare data
%cd data
!python prepare_voz_balanced.py
%cd ..

# Train
!python src/train.py
```

## 🐛 Troubleshooting

### "CUDA out of memory"
```python
# Edit src/train.py
BATCH_TRAIN = 24  # Reduce from 32
BATCH_EVAL = 48   # Reduce from 64
```

### "Download too slow"
```bash
# Use Hugging Face cache
export HF_HOME=/path/to/cache
python data/prepare_voz_balanced.py
```

### "Training seems slow"
```bash
# Check GPU usage
nvidia-smi
# Should see 80-90% GPU utilization, 12-13GB VRAM
```

See [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) for more troubleshooting.

## 📈 Performance Benchmarks

**Training Speed (T4 GPU):**
- Data preparation: 5-10 min (one-time)
- Training (547K): ~1.5 hours
- Evaluation: ~30 seconds
- Total: ~1.6 hours

**Resource Usage:**
- VRAM: 12-13GB / 16GB (80%+ utilization)
- CPU: 2 workers for data loading
- Disk: ~2GB (dataset + model)

**Model Performance:**
- F1 Score: 75-80% (balanced dataset)
- Precision: 75-80% (good for UX)
- Recall: 75-80% (catches most hate speech)
- Production: Tune threshold (0.5 → 0.6 for better precision)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Support for other Vietnamese BERT models (viBERT, viT5)
- [ ] Multi-class classification (clean, offensive, hate)
- [ ] Real-time inference optimization (ONNX export)
- [ ] Docker containerization
- [ ] FastAPI inference server

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- PhoBERT model: [VinAI Research](https://github.com/VinAIResearch/PhoBERT)
- VOZ-HSD dataset: [tarudesu/VOZ-HSD](https://huggingface.co/datasets/tarudesu/VOZ-HSD)
- Paper: [ViHateT5 @ ACL 2024](https://aclanthology.org/2024.findings-acl.355)

---

**⭐ Star this repo if you find it useful!**
