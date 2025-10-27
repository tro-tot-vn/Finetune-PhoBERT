# Fix Progress Bar Spam in Colab

## 🐛 Problem

Progress bars in Colab tạo NHIỀU DÒNG mới thay vì update 1 dòng → lag notebook!

---

## ✅ Solutions (Ordered by preference)

### **Solution 1: Automatic (Already in code)** ⭐

Code đã tự động detect Colab và giảm update frequency:

```python
# In train.py - automatically applied
os.environ['TQDM_MININTERVAL'] = '10'  # Update every 10 seconds
```

**Expected behavior:**
- Progress bar updates every 10 seconds (not every step)
- Much less spam
- Still see progress

---

### **Solution 2: Manual - Disable Progress Bars**

Nếu vẫn lag, run TRƯỚC khi train:

```python
# In Colab cell, run BEFORE training
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TQDM_DISABLE'] = '1'  # Disable all progress bars

# Then train
!python src/train.py
```

**Result:**
- ❌ No progress bars
- ✅ Only text logs every 100 steps
- ✅ No lag

**Output:**
```
Step 100: loss=0.45
Step 200: loss=0.42
Step 300: loss=0.40
...
```

---

### **Solution 3: Use Logging Only Mode**

Edit `src/train.py` line 242:

```python
# Change from:
disable_tqdm=False,

# To:
disable_tqdm=True,  # No progress bars, logs only
```

**Result:**
- Clean text-only output
- Logs every 100 steps
- Perfect for Colab

---

### **Solution 4: Increase Update Interval** (Already applied)

Already in code, but you can adjust:

```python
# In train.py line 150-151
os.environ['TQDM_MININTERVAL'] = '30'  # Update every 30 seconds (slower)
os.environ['TQDM_MAXINTERVAL'] = '60'  # Max 60 seconds
```

---

## 🎯 Recommended Workflow

### **Try in order:**

1. ✅ **Run with default** (auto-configured, updates every 10s)
   ```python
   !python src/train.py
   ```

2. ⚠️ **If still laggy**, disable tqdm before running:
   ```python
   import os
   os.environ['TQDM_DISABLE'] = '1'
   !python src/train.py
   ```

3. 🔧 **If want permanent fix**, edit train.py:
   ```python
   # Line 242
   disable_tqdm=True,
   ```

---

## 📊 Expected Output (Clean Mode)

### **With Progress Bars (updates every 10s):**
```
[Colab] Progress bars configured for Jupyter/Colab (updates every 10s)
[GPU] Tesla T4 (Compute Capability: 7.5)

Training: 7% 1000/13656 [02:15<28:45, 7.33it/s]
Training: 15% 2000/13656 [04:30<26:30, 7.33it/s]
...
```

### **Without Progress Bars (cleanest):**
```
[GPU] Tesla T4 (Compute Capability: 7.5)

Step 100: {'loss': 0.45, 'learning_rate': 1.9e-05, 'epoch': 0.07}
Step 200: {'loss': 0.42, 'learning_rate': 1.8e-05, 'epoch': 0.15}
Step 300: {'loss': 0.40, 'learning_rate': 1.7e-05, 'epoch': 0.22}
...

Epoch 1 complete: {'eval_loss': 0.478, 'eval_f1_inv': 0.731}
```

---

## 🚀 Quick Test

Test which works best for you:

```python
# Option A: Default (auto-config)
!python src/train.py

# Option B: Force disable tqdm
!TQDM_DISABLE=1 python src/train.py

# Option C: Very slow updates (60s)
!TQDM_MININTERVAL=60 python src/train.py
```

Pick what works best and stick with it! ✅

