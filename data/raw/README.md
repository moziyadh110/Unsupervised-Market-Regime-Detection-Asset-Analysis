# Raw Data Directory

Place your raw CSV data files here.

## Expected Files

- `merged_file_FINAL.csv` - Your original merged dataset
- Or any other raw data files from your sources

## Usage

```python
from src.data_loader import load_csv

df = load_csv('data/raw/your_file.csv')
```

## Note

Raw data files are excluded from Git (see `.gitignore`).
Store large files using Git LFS if needed.
