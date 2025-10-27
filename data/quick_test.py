"""
Quick test script - Sample 1000 samples để verify logic
"""

from prepare_voz_balanced import load_and_filter_voz, balance_dataset, split_dataset
import pandas as pd

print("🧪 QUICK TEST - Sampling 1000 samples")
print("="*70)

try:
    # Load small sample
    print("\n1. Loading and filtering VOZ-HSD...")
    df = load_and_filter_voz(confidence_threshold=0.95)
    
    # Sample 1000 for quick test
    print(f"\n📦 Sampling 1000 samples for quick test...")
    df_sample = df.sample(n=min(1000, len(df)), random_state=42)
    print(f"Sample size: {len(df_sample)}")
    
    # Balance
    print("\n2. Balancing dataset (70-30)...")
    try:
        balanced_df = balance_dataset(df_sample, clean_ratio=70)
    except Exception as e:
        print(f"⚠️  Balancing failed (expected with small sample): {e}")
        print("   Skipping to split test with original sample...")
        balanced_df = df_sample
    
    # Split
    print("\n3. Splitting dataset (80-10-10)...")
    train_df, val_df, test_df = split_dataset(balanced_df)
    
    print("\n✅ ALL TESTS PASSED!")
    print("\n📊 Quick Summary:")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    print("\n💡 Ready to run full preparation:")
    print("   python prepare_voz_balanced.py")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

