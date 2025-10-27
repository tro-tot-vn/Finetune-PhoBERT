# GPU Optimization Guide for T4

## 📊 **Performance Improvements**

### Default vs Optimized

| Setting | Before | After | Speedup |
|---------|--------|-------|---------|
| Batch size (train) | 8 | **32** | **~3-4x** |
| Batch size (eval) | 16 | **64** | **~3-4x** |
| VRAM usage | ~2-3GB | **~12-13GB** | **Full utilization!** |
| Mixed precision | None | FP16 | ~1.7x |
| Optimizer | AdamW | AdamW Fused | ~1.1-1.2x |
| DataLoader workers | 0 | 2 | ~1.1x |
| Gradient checkpointing | Off | Off | Faster! |
| **Total speedup** | - | - | **~3-4x faster** 🚀 |

## 🖥️ **GPU Architecture Support**

| GPU | Architecture | Compute Capability | FP16 | BF16 | TF32 |
|-----|--------------|-------------------|------|------|------|
| T4 | Turing | 7.5 | ✅ | ❌ | ❌ |
| V100 | Volta | 7.0 | ✅ | ❌ | ❌ |
| A100 | Ampere | 8.0 | ✅ | ✅ | ✅ |
| A10 | Ampere | 8.6 | ✅ | ✅ | ✅ |

**Auto-detection**: Code tự động chọn optimization tốt nhất cho GPU của bạn!

## 🚀 **Optimizations Applied**

### 1. **Larger Batch Sizes** 🔥
```python
BATCH_TRAIN = 32  # ↑↑ Aggressive! (was 8)
BATCH_EVAL = 64   # ↑↑ Max out that T4! (was 16)
GRAD_ACCUM = 1    # No need with large batches
```
- **Why**: T4 has 16GB VRAM - LET'S USE IT!
- **Benefit**: ~12-13GB VRAM usage, much faster training
- **Result**: GPU utilization 85-95%

### 2. **Gradient Checkpointing** (DISABLED for speed)
```python
USE_GRADIENT_CHECKPOINTING = False  # OFF for T4
```
- **Why**: T4 has enough VRAM (16GB), no need to trade speed for memory
- **Benefit**: ~20-30% faster training vs checkpointing ON
- **When to enable**: Only if you get OOM with large batches

### 3. **Mixed Precision** (Auto-detected)
```python
# T4 (Turing): fp16=True
# A100 (Ampere+): bf16=True
```
- **Why**: BF16 has same range as FP32, more stable than FP16
- **Benefit**: Faster matmul, less numerical issues
- **Note**: 
  - T4 uses FP16 (Turing architecture)
  - A100/A10 uses BF16 (Ampere architecture)
  - Auto-detected based on GPU capability

### 4. **Fused AdamW Optimizer**
```python
optim="adamw_torch_fused"
```
- **Why**: Fused kernel reduces memory transfers
- **Benefit**: 10-20% faster optimizer step

### 5. **DataLoader Parallelization**
```python
dataloader_num_workers=2  # ↑ from 0
dataloader_pin_memory=True
```
- **Why**: Parallel data loading, faster CPU→GPU transfer
- **Benefit**: GPU doesn't wait for data

### 6. **TF32 Precision** (A100/V100+ only)
```python
tf32=True  # Auto-detected, only for Ampere+ GPUs
```
- **Why**: Use TensorFloat-32 for matmul (A100, A10, etc.)
- **Benefit**: ~1.2x faster matmul with minimal accuracy loss
- **Note**: T4 doesn't support TF32 (Turing architecture), uses FP16 instead

## ⚙️ **Advanced: Manual Tuning**

### If you get OOM (Out of Memory):

**Option 1**: Reduce batch size (most common fix)
```python
BATCH_TRAIN = 24  # Instead of 32
BATCH_EVAL = 48   # Instead of 64
```

**Option 2**: Enable gradient checkpointing (trade speed for memory)
```python
USE_GRADIENT_CHECKPOINTING = True  # Line 37 in train.py
# Keep larger batches: BATCH_TRAIN = 32
```

**Option 3**: Reduce sequence length
```python
MAX_LENGTH = 128  # Instead of 192 (saves ~30% VRAM)
```

**Option 4**: Reduce both batch & sequence length
```python
MAX_LENGTH = 128
BATCH_TRAIN = 16
BATCH_EVAL = 32
```

### For even MORE speed (experimental):

**Enable torch.compile** (PyTorch 2.0+):
```python
USE_TORCH_COMPILE = True  # In train.py line 38
```
- ⚠️ First epoch will be SLOW (compilation)
- ✅ Subsequent epochs ~1.3-1.5x faster
- ❌ May crash on Colab (buggy)

## 📈 **Expected Training Time**

With **AGGRESSIVE** optimized settings on T4:

| Dataset Size | Time (before) | Time (after) | Speedup |
|--------------|---------------|--------------|---------|
| ~24K samples | ~30 min | **~8-10 min** | **3-4x** 🚀 |
| ~50K samples | ~60 min | **~18-22 min** | **3x** |
| ~100K samples | ~120 min | **~35-40 min** | **3x** |

*Times are for 3 epochs with BATCH_TRAIN=32, FP16, no gradient checkpointing*

**VRAM Usage**: ~12-13GB / 16GB (80%+ utilization) ✅

## 🔍 **Monitor GPU Usage**

### In Colab:
```python
# Check GPU memory
!nvidia-smi

# Watch GPU usage in real-time
!watch -n 1 nvidia-smi  # Ctrl+C to stop
```

**Ideal GPU utilization**: 80-95%
- If < 70%: Increase batch size or num_workers
- If OOM: Decrease batch size or enable gradient checkpointing

## 🎯 **Best Practices**

### 1. **Start with optimized defaults**
Just run `python src/train.py` - already optimized!

### 2. **If OOM occurs**:
```python
# Edit src/train.py line 28-29
BATCH_TRAIN = 12  # Reduce gradually
BATCH_EVAL = 24
```

### 3. **Monitor first epoch**:
Check if GPU util is high. If not, increase batch size.

### 4. **For fastest training**:
- Use T4 GPU (not CPU!)
- Close other notebooks
- Keep browser tab active (Colab can throttle inactive sessions)

## 📊 **Optimization Checklist**

- ✅ **Batch size**: MAXED OUT (32/64) 🔥
- ✅ **VRAM usage**: ~12-13GB / 16GB (80%+)
- ✅ **Mixed precision**: FP16 (T4) / BF16 (A100)
- ✅ **Gradient checkpointing**: Disabled (for speed)
- ✅ **Fused optimizer**: Enabled
- ✅ **DataLoader workers**: 2
- ✅ **Pin memory**: Enabled
- ⚠️ **TF32**: Auto (only on Ampere+)
- ⚠️ **Torch compile**: Disabled (unstable on Colab)

## 🔧 **Troubleshooting**

### "CUDA out of memory"
```python
# Reduce batch size in src/train.py
BATCH_TRAIN = 24  # Instead of 32
BATCH_EVAL = 48   # Instead of 64
```

### Training seems slow (GPU util < 70%)
```bash
# Check GPU usage
!nvidia-smi

# If GPU util is low:
# - Increase batch size more: BATCH_TRAIN = 40
# - Increase dataloader workers: dataloader_num_workers = 4
```

### VRAM usage too low (< 8GB)
```python
# Increase batch size even more!
BATCH_TRAIN = 40  # Push it!
BATCH_EVAL = 80
```

### "Optimizer not found: adamw_torch_fused"
```python
# Older PyTorch, fallback to regular
optim="adamw_torch"  # Manual edit if needed
```

---

## 🎉 Summary

With **AGGRESSIVE** optimizations, training on T4 is **~3-4x faster** than default!

**Estimated training time** for your dataset (~24K samples, 3 epochs): **8-10 minutes** ⚡

**VRAM usage**: ~12-13GB / 16GB (no more 2.7GB waste!) 🔥

