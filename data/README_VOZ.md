# VOZ-HSD Data Preparation Guide

## 🎯 Quick Start

### Default (70-30, probs≥0.95):
```bash
cd data
python prepare_voz_balanced.py
```

**Output:**
- `train.csv` (~437K samples, 80%)
- `valid.csv` (~55K samples, 10%)
- `test.csv` (~55K samples, 10%)
- `dataset_summary.txt` (detailed report)

**Expected size:** ~547K total samples

---

## ⚙️ Configuration Options

### 1. Change Confidence Threshold

```bash
# More strict (higher quality, fewer samples)
python prepare_voz_balanced.py --confidence 0.98

# Less strict (more samples, some noise)
python prepare_voz_balanced.py --confidence 0.90
```

### 2. Change Balance Ratio

```bash
# 80-20 (closer to real-world)
python prepare_voz_balanced.py --ratio 80

# 60-40 (stronger balance)
python prepare_voz_balanced.py --ratio 60

# 50-50 (perfect balance)
python prepare_voz_balanced.py --ratio 50
```

### 3. Change Split Ratios

```bash
# 70-15-15 split
python prepare_voz_balanced.py --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

# 90-5-5 split (more training data)
python prepare_voz_balanced.py --train_ratio 0.9 --val_ratio 0.05 --test_ratio 0.05
```

### 4. Custom Output Directory

```bash
python prepare_voz_balanced.py --output_dir experiments/voz_70_30/
```

---

## 📊 Recommended Configurations

### For UX-focused (Precision priority):
```bash
python prepare_voz_balanced.py --confidence 0.95 --ratio 80
```
- High quality labels
- Closer to real-world distribution
- Expected F1: 73-77%, Precision: 76-80%

### For Balanced performance (Default):
```bash
python prepare_voz_balanced.py --confidence 0.95 --ratio 70
```
- Good balance between precision & recall
- Use all minority class samples
- Expected F1: 75-80%

### For Recall-focused (Safety priority):
```bash
python prepare_voz_balanced.py --confidence 0.90 --ratio 60
```
- More data, some noise acceptable
- Higher recall for minority class
- Expected F1: 76-82%, Recall: 82-87%

---

## 📈 Expected Dataset Sizes

| Confidence | Ratio | Total Samples | Train | Val | Test |
|------------|-------|---------------|-------|-----|------|
| **0.95** | **70-30** | **547K** | 437K | 55K | 55K |
| 0.95 | 80-20 | 820K | 656K | 82K | 82K |
| 0.95 | 60-40 | 410K | 328K | 41K | 41K |
| 0.90 | 70-30 | 795K | 636K | 80K | 80K |
| 0.98 | 70-30 | 320K | 256K | 32K | 32K |

---

## 🚀 Full Workflow

```bash
# Step 1: Prepare data (5-10 minutes)
cd data
python prepare_voz_balanced.py --confidence 0.95 --ratio 70

# Step 2: Check summary
cat dataset_summary.txt

# Step 3: Train model (1-2 hours)
cd ..
python src/train.py

# Step 4: Evaluate
python src/evaluation.py
python src/plot_training.py
```

---

## 📋 A/B Testing Multiple Configurations

### Test Script:
```bash
#!/bin/bash
# test_configurations.sh

# Config 1: 80-20
python prepare_voz_balanced.py --confidence 0.95 --ratio 80 --output_dir data_80_20/

# Config 2: 70-30 (default)
python prepare_voz_balanced.py --confidence 0.95 --ratio 70 --output_dir data_70_30/

# Config 3: 60-40
python prepare_voz_balanced.py --confidence 0.95 --ratio 60 --output_dir data_60_40/

echo "✅ All configurations prepared!"
echo "Now train models with each config and compare results."
```

### Compare Results:
```bash
# Train on each config
for config in 80_20 70_30 60_40; do
    cp -r data_${config}/* data/
    python src/train.py
    mv out/metrics out_${config}
done

# Compare F1 scores
echo "Config | F1 Score | Precision | Recall"
for config in 80_20 70_30 60_40; do
    python -c "import json; d=json.load(open('out_${config}/report.json')); print(f'${config} | {d[\"f1_inv\"]:.3f} | {d[\"precision_inv\"]:.3f} | {d[\"recall_inv\"]:.3f}')"
done
```

---

## 🔍 Troubleshooting

### Issue: "Out of memory"
```bash
# Use smaller dataset
python prepare_voz_balanced.py --confidence 0.98 --ratio 70
# Or increase confidence to get fewer samples
```

### Issue: "Not enough clean samples"
```bash
# Lower the ratio
python prepare_voz_balanced.py --ratio 60
# Or lower confidence threshold
python prepare_voz_balanced.py --confidence 0.90
```

### Issue: Download too slow
```bash
# Use cache if available
export HF_HOME=/path/to/cache
python prepare_voz_balanced.py
```

---

## 📊 Dataset Statistics

Based on VOZ-HSD @ probs ≥ 0.95:

```
Available samples:
- Clean: 8,574,943 (98.12%)
- Hate: 163,987 (1.88%)

After 70-30 balancing:
- Clean: 382,304 (70%)
- Hate: 163,987 (30%)
- Total: 546,291

After 80-10-10 split:
- Train: 437,033 (80%)
- Val: 54,629 (10%)
- Test: 54,629 (10%)
```

---

## ⚠️ Important Notes

1. **Download time**: First run downloads ~1.2GB from Hugging Face (5-10 min)
2. **Processing time**: Filtering & sampling ~2-3 minutes
3. **Disk space**: Output CSVs ~500MB-1GB depending on config
4. **Random seed**: Fixed to 42 for reproducibility
5. **Stratification**: All splits maintain class balance

---

## 📚 References

- VOZ-HSD Paper: [ViHateT5 @ ACL 2024](https://aclanthology.org/2024.findings-acl.355)
- Dataset: [tarudesu/VOZ-HSD](https://huggingface.co/datasets/tarudesu/VOZ-HSD)

---

## 🆘 Support

If you encounter issues:
1. Check `dataset_summary.txt` for details
2. Verify you have internet connection (for download)
3. Ensure you have ~2GB free disk space
4. Check Python version (≥3.8)

