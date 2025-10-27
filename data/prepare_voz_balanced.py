"""
VOZ-HSD Data Processor
Prepare balanced dataset (70-30) with high-quality labels for Vietnamese hate speech detection.

Usage:
    python prepare_voz_balanced.py --confidence 0.95 --ratio 70 --output_dir data/
"""

import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def load_and_filter_voz(confidence_threshold=0.95):
    """
    Load VOZ-HSD dataset và filter theo confidence threshold.
    
    Args:
        confidence_threshold: Minimum confidence score (0.0-1.0)
    
    Returns:
        pandas DataFrame với columns: text, label
    """
    print(f"\n{'='*70}")
    print("🔄 LOADING VOZ-HSD DATASET")
    print(f"{'='*70}")
    
    print("Downloading from Hugging Face...")
    dataset = load_dataset("tarudesu/VOZ-HSD", split="train")
    print(f"✅ Loaded {len(dataset):,} samples")
    
    print(f"\n🔍 Filtering samples with probs >= {confidence_threshold}...")
    
    # Convert to pandas for easier processing
    df = pd.DataFrame({
        'text': dataset['texts'],
        'label': dataset['labels'],
        'probs': dataset['probs']
    })
    
    print(f"Original size: {len(df):,}")
    
    # Filter by confidence
    filtered_df = df[df['probs'] >= confidence_threshold].copy()
    print(f"After filtering (probs >= {confidence_threshold}): {len(filtered_df):,}")
    
    # Show label distribution
    label_dist = filtered_df['label'].value_counts()
    print(f"\nLabel distribution after filtering:")
    print(f"  Label 0 (clean):   {label_dist[0]:,} ({label_dist[0]/len(filtered_df)*100:.2f}%)")
    print(f"  Label 1 (hate):    {label_dist[1]:,} ({label_dist[1]/len(filtered_df)*100:.2f}%)")
    
    # Drop probs column (no longer needed)
    filtered_df = filtered_df[['text', 'label']]
    
    return filtered_df


def balance_dataset(df, clean_ratio=70):
    """
    Balance dataset theo ratio mong muốn.
    
    Args:
        df: Input DataFrame
        clean_ratio: % của clean samples (0-100), hate sẽ là 100-clean_ratio
    
    Returns:
        Balanced DataFrame
    """
    print(f"\n{'='*70}")
    print(f"⚖️  BALANCING DATASET ({clean_ratio}-{100-clean_ratio})")
    print(f"{'='*70}")
    
    # Separate by class
    clean_df = df[df['label'] == 0].copy()
    hate_df = df[df['label'] == 1].copy()
    
    print(f"Available samples:")
    print(f"  Clean: {len(clean_df):,}")
    print(f"  Hate:  {len(hate_df):,}")
    
    # Take ALL hate samples
    n_hate = len(hate_df)
    
    # Calculate required clean samples for desired ratio
    # If ratio is 70-30: clean/hate = 70/30 = 2.333
    # So: n_clean = n_hate * (clean_ratio / hate_ratio)
    hate_ratio = 100 - clean_ratio
    n_clean = int(n_hate * (clean_ratio / hate_ratio))
    
    if n_clean > len(clean_df):
        print(f"\n⚠️  WARNING: Not enough clean samples!")
        print(f"   Requested: {n_clean:,}")
        print(f"   Available: {len(clean_df):,}")
        print(f"   Using all available clean samples instead.")
        n_clean = len(clean_df)
    
    print(f"\nTarget balanced dataset:")
    print(f"  Clean: {n_clean:,} ({n_clean/(n_clean+n_hate)*100:.1f}%)")
    print(f"  Hate:  {n_hate:,} ({n_hate/(n_clean+n_hate)*100:.1f}%)")
    print(f"  Total: {n_clean + n_hate:,}")
    
    # Sample clean
    clean_sampled = clean_df.sample(n=n_clean, random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([clean_sampled, hate_df], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Balanced dataset created: {len(balanced_df):,} samples")
    
    return balanced_df


def split_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset thành train/val/test với stratification.
    
    Args:
        df: Input DataFrame
        train_ratio: Tỉ lệ train set (0.0-1.0)
        val_ratio: Tỉ lệ validation set (0.0-1.0)
        test_ratio: Tỉ lệ test set (0.0-1.0)
    
    Returns:
        train_df, val_df, test_df
    """
    print(f"\n{'='*70}")
    print(f"✂️  SPLITTING DATASET ({train_ratio*100:.0f}-{val_ratio*100:.0f}-{test_ratio*100:.0f})")
    print(f"{'='*70}")
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=df['label']
    )
    
    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=42,
        stratify=temp_df['label']
    )
    
    # Print statistics
    def print_split_stats(name, split_df, total):
        n = len(split_df)
        n_clean = (split_df['label'] == 0).sum()
        n_hate = (split_df['label'] == 1).sum()
        print(f"\n{name}:")
        print(f"  Total: {n:,} ({n/total*100:.1f}%)")
        print(f"  Clean: {n_clean:,} ({n_clean/n*100:.1f}%)")
        print(f"  Hate:  {n_hate:,} ({n_hate/n*100:.1f}%)")
    
    total = len(df)
    print_split_stats("📘 Train", train_df, total)
    print_split_stats("📗 Validation", val_df, total)
    print_split_stats("📕 Test", test_df, total)
    
    return train_df, val_df, test_df


def save_datasets(train_df, val_df, test_df, output_dir):
    """
    Lưu datasets vào CSV files.
    
    Args:
        train_df, val_df, test_df: DataFrames to save
        output_dir: Output directory path
    """
    print(f"\n{'='*70}")
    print("💾 SAVING DATASETS")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'valid.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Saved:")
    print(f"   {train_path} ({len(train_df):,} samples)")
    print(f"   {val_path} ({len(val_df):,} samples)")
    print(f"   {test_path} ({len(test_df):,} samples)")


def generate_summary(train_df, val_df, test_df, confidence, ratio, output_dir):
    """
    Generate summary report.
    """
    summary_path = os.path.join(output_dir, 'dataset_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("VOZ-HSD DATASET PREPARATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  - Source: VOZ-HSD (Hugging Face)\n")
        f.write(f"  - Confidence threshold: {confidence}\n")
        f.write(f"  - Balance ratio: {ratio}-{100-ratio} (clean-hate)\n")
        f.write(f"  - Split: 80-10-10 (train-val-test)\n\n")
        
        total = len(train_df) + len(val_df) + len(test_df)
        
        f.write(f"Dataset Sizes:\n")
        f.write(f"  - Total: {total:,} samples\n")
        f.write(f"  - Train: {len(train_df):,} ({len(train_df)/total*100:.1f}%)\n")
        f.write(f"  - Validation: {len(val_df):,} ({len(val_df)/total*100:.1f}%)\n")
        f.write(f"  - Test: {len(test_df):,} ({len(test_df)/total*100:.1f}%)\n\n")
        
        for name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            n_clean = (df['label'] == 0).sum()
            n_hate = (df['label'] == 1).sum()
            f.write(f"{name} Distribution:\n")
            f.write(f"  - Clean: {n_clean:,} ({n_clean/len(df)*100:.1f}%)\n")
            f.write(f"  - Hate:  {n_hate:,} ({n_hate/len(df)*100:.1f}%)\n\n")
        
        f.write("Files Generated:\n")
        f.write(f"  - train.csv\n")
        f.write(f"  - valid.csv\n")
        f.write(f"  - test.csv\n")
        f.write(f"  - dataset_summary.txt (this file)\n")
    
    print(f"\n📄 Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare balanced VOZ-HSD dataset')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence threshold (default: 0.95)')
    parser.add_argument('--ratio', type=int, default=70,
                       help='Clean ratio percentage (default: 70 for 70-30)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory (default: /)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train split ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test split ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 VOZ-HSD DATA PROCESSOR")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Confidence threshold: {args.confidence}")
    print(f"  - Balance ratio: {args.ratio}-{100-args.ratio} (clean-hate)")
    print(f"  - Train-Val-Test: {args.train_ratio*100:.0f}-{args.val_ratio*100:.0f}-{args.test_ratio*100:.0f}")
    print(f"  - Output directory: {args.output_dir}")
    
    try:
        # Step 1: Load and filter
        df = load_and_filter_voz(args.confidence)
        
        # Step 2: Balance
        balanced_df = balance_dataset(df, args.ratio)
        
        # Step 3: Split
        train_df, val_df, test_df = split_dataset(
            balanced_df, 
            args.train_ratio, 
            args.val_ratio, 
            args.test_ratio
        )
        
        # Step 4: Save
        save_datasets(train_df, val_df, test_df, args.output_dir)
        
        # Step 5: Generate summary
        generate_summary(train_df, val_df, test_df, args.confidence, args.ratio, args.output_dir)
        
        print(f"\n{'='*70}")
        print("✅ SUCCESS! Dataset preparation complete.")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {str(e)}")
        print(f"{'='*70}\n")
        raise


if __name__ == "__main__":
    main()

