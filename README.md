# Finetune-PhoBERT

Fine-tuning PhoBERT cho Vietnamese text moderation (binary classification: valid/invalid).

## 📦 Features

- ✅ Fine-tune PhoBERT-base-v2 cho binary classification
- ✅ Weighted loss cho imbalanced data
- ✅ Early stopping với best checkpoint selection
- ✅ Comprehensive training reports & visualizations
- ✅ Multiple evaluation metrics (ROC, PR curve, threshold analysis)
- ⚡ GPU optimized (2x faster on T4: ~12-15 min for 24K samples)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model
cd src && python train.py

# 3. Visualize training
python plot_training.py

# 4. Evaluate on test set
python evaluation.py
```

## 📊 Outputs

Training sẽ tạo:
- **Model**: `models/phobert-moderation/`
- **Training report**: `out/metrics/training_report.md`
- **Visualizations**: `out/metrics/*.png` (8 plots)
- **Metrics**: `out/metrics/*.json`

Chi tiết xem [USAGE.md](USAGE.md)

## 📁 Structure

```
.
├── data/
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   └── normalize_labels_flex.py
├── src/
│   ├── train.py              # Training script
│   ├── evaluation.py         # Evaluation + visualization
│   ├── callbacks.py          # Custom callbacks
│   ├── report_generator.py   # Report generation
│   └── plot_training.py      # Training curves
└── requirements.txt
```

## 🔧 Configuration

Edit hyperparameters in `src/train.py`:
- Model: `vinai/phobert-base-v2`
- Max length: 192 tokens
- Batch size: 8 (train) / 16 (eval)
- Epochs: 3
- Learning rate: 2e-5

## 📝 For Research Papers

See [USAGE.md](USAGE.md) for detailed instructions on:
- Extracting metrics for LaTeX tables
- Selecting visualizations for papers
- Understanding output files

## License

MIT
