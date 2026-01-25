# KBBI Database Setup

This folder contains the KBBI (Kamus Besar Bahasa Indonesia) database used for dual verification to filter false positives from the ML model.

## Database Information

- **Source**: [dyazincahya/KBBI-SQL-database](https://github.com/dyazincahya/KBBI-SQL-database)
- **Total Words**: 115,978 Indonesian words
- **File Size**: ~31 MB
- **Format**: JSON

## Setup Instructions

### Option 1: Automatic Download (Recommended)

Run the setup script to automatically download the KBBI database:

```bash
# From project root directory
bash scripts/setup_kbbi.sh
```

### Option 2: Manual Download

1. Download the file directly from GitHub:
   ```bash
   cd src/data/
   curl -L -o dictionary_JSON.json "https://raw.githubusercontent.com/dyazincahya/KBBI-SQL-database/master/dictionary_JSON.json"
   ```

2. Or download from browser:
   - Visit: https://github.com/dyazincahya/KBBI-SQL-database
   - Download `dictionary_JSON.json`
   - Place it in `src/data/dictionary_JSON.json`

### Verification

After download, verify the file exists:

```bash
ls -lh src/data/dictionary_JSON.json
# Should show: ~31M dictionary_JSON.json
```

## Database Structure

The JSON file contains:

```json
{
  "dictionary": [
    {
      "_id": 11,
      "word": "kata",
      "arti": "definisi...",
      "type": 2
    },
    ...
  ]
}
```

## Usage

The KBBI database is automatically loaded when the FastAPI backend starts. It's used to:

1. **Filter False Positives**: ML model may incorrectly detect Indonesian words as foreign. KBBI verification removes these.
2. **Dual Verification**: Combines ML prediction with dictionary lookup for higher accuracy.
3. **Zero API Calls**: All verification happens locally (instant, no rate limits).

## Performance

- **Load Time**: ~1-2 seconds at startup
- **Memory Usage**: ~10-20 MB (words loaded into Python set)
- **Lookup Speed**: O(1) instant verification per word

## Troubleshooting

### File Not Found Error

If you see:
```
FileNotFoundError: KBBI database not found at: src/data/dictionary_JSON.json
```

Solution: Download the file using one of the methods above.

### Invalid JSON Error

If the file is corrupted, re-download it:
```bash
rm src/data/dictionary_JSON.json
bash scripts/setup_kbbi.sh
```

## License

KBBI database is provided by [dyazincahya/KBBI-SQL-database](https://github.com/dyazincahya/KBBI-SQL-database).

The original KBBI content is from [KBBI Daring Kemdikbud](https://kbbi.kemdikbud.go.id/).
