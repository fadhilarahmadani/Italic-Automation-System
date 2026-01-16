# src/split_dataset.py
import json
import random
from pathlib import Path

def split_dataset(input_path, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split dataset menjadi train, validation, dan test
    
    Args:
        input_path: Path ke dataset JSON
        output_dir: Folder output
        train_ratio: Proporsi data training (default 0.8 = 80%)
        val_ratio: Proporsi data validation (default 0.1 = 10%)
        seed: Random seed untuk reproducibility
    """
    # Load dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Shuffle data
    random.seed(seed)
    random.shuffle(data)
    
    # Hitung jumlah data per split
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size
    
    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Buat output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Simpan splits
    splits = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        output_path = f"{output_dir}/{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"✅ {split_name}: {len(split_data)} samples → {output_path}")
    
    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Total: {total}")
    print(f"   Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"   Validation: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"   Test: {len(test_data)} ({len(test_data)/total*100:.1f}%)")

if __name__ == "__main__":
    split_dataset(
        input_path="data/italic_dataset.json",
        output_dir="data",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )
