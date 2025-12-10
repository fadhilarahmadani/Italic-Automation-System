# src/validate_dataset.py
import json

def validate_dataset(path):
    """Validasi format dataset"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Total data: {len(data)}")
    
    errors = []
    for i, item in enumerate(data):
        # Cek key wajib
        if not all(k in item for k in ['id', 'tokens', 'labels']):
            errors.append(f"Data {i+1}: Missing required keys")
            continue
        
        # Cek panjang tokens == labels
        if len(item['tokens']) != len(item['labels']):
            errors.append(f"Data {i+1}: Tokens length ({len(item['tokens'])}) != Labels length ({len(item['labels'])})")
        
        # Cek label hanya O, B, I
        valid_labels = {'O', 'B', 'I'}
        invalid = set(item['labels']) - valid_labels
        if invalid:
            errors.append(f"Data {i+1}: Invalid labels: {invalid}")
    
    if errors:
        print("\n❌ Dataset memiliki error:")
        for err in errors:
            print(f"   - {err}")
        return False
    else:
        print("✅ Dataset valid!")
        
        # Statistik
        total_tokens = sum(len(item['tokens']) for item in data)
        total_b = sum(item['labels'].count('B') for item in data)
        total_i = sum(item['labels'].count('I') for item in data)
        total_o = sum(item['labels'].count('O') for item in data)
        
        print(f"\n📈 Statistik:")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Label B: {total_b} ({total_b/total_tokens*100:.1f}%)")
        print(f"   Label I: {total_i} ({total_i/total_tokens*100:.1f}%)")
        print(f"   Label O: {total_o} ({total_o/total_tokens*100:.1f}%)")
        
        return True

if __name__ == "__main__":
    validate_dataset("data/italic_dataset.json")
