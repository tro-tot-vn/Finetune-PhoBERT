"""
Debug training speed bottlenecks
"""
import time
import torch
from src.train import *

print("="*70)
print("🔍 SPEED DIAGNOSTIC")
print("="*70)

# Load data
print("\n1. Loading data...")
ds, train_df, _, _ = read_split(DATA_DIR)
print(f"✅ Loaded {len(train_df)} training samples")

# Tokenizer
print("\n2. Building tokenizer...")
start = time.time()
tokenizer = build_tokenizer()
print(f"✅ Tokenizer loaded in {time.time()-start:.2f}s")

# Encoding
print("\n3. Encoding datasets...")
start = time.time()
enc = encode_datasets(ds, tokenizer)
encoding_time = time.time() - start
print(f"✅ Encoding took {encoding_time:.2f}s ({len(train_df)/encoding_time:.0f} samples/sec)")

# DataLoader
print("\n4. Testing DataLoader speed...")
collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

from torch.utils.data import DataLoader
train_loader = DataLoader(
    enc["train"],
    batch_size=96,
    collate_fn=collator,
    num_workers=0,
    pin_memory=True
)

print(f"   Testing 10 batches...")
start = time.time()
for i, batch in enumerate(train_loader):
    if i >= 10:
        break
batch_time = (time.time() - start) / 10
print(f"✅ Average batch loading: {batch_time:.3f}s per batch")
print(f"   Expected speed: {1/batch_time:.2f} batches/sec")

# GPU inference
print("\n5. Testing GPU forward pass...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

# Test forward pass
batch = next(iter(train_loader))
batch = {k: v.to(device) for k, v in batch.items()}

start = time.time()
for _ in range(10):
    with torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
forward_time = (time.time() - start) / 10
print(f"✅ Average forward+backward: {forward_time:.3f}s")
print(f"   Expected speed: {1/forward_time:.2f} it/sec")

# Summary
print("\n" + "="*70)
print("📊 BOTTLENECK ANALYSIS")
print("="*70)
print(f"DataLoader time:  {batch_time:.3f}s per batch")
print(f"GPU compute time: {forward_time:.3f}s per batch")
print(f"Total per step:   {batch_time + forward_time:.3f}s")
print(f"Expected speed:   {1/(batch_time + forward_time):.2f} it/s")

if batch_time > forward_time * 2:
    print("\n⚠️  BOTTLENECK: DataLoader (CPU/disk I/O slow)")
    print("   Solutions:")
    print("   - Increase tokenization batch_size")
    print("   - Ensure data is cached")
    print("   - Check disk speed")
elif forward_time > batch_time * 2:
    print("\n⚠️  BOTTLENECK: GPU computation")
    print("   Solutions:")
    print("   - Reduce batch size")
    print("   - Check GPU utilization")
else:
    print("\n✅ Balanced - no major bottleneck")
    print(f"   Should achieve ~{1/(batch_time + forward_time):.1f} it/s in training")

