# Quick Start Guide

## 5-Minute Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/geopolitical-risk-analysis.git
cd geopolitical-risk-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook notebooks/
```

## Running Notebooks in Sequence

1. **01_data_collection.ipynb** - Load raw data
2. **02_data_cleaning.ipynb** - Clean & prepare
3. **03_eda_analysis.ipynb** - Exploratory analysis
4. **04_feature_engineering.ipynb** - Create features
5. **05_ml_clustering.ipynb** - Train models
6. **06_model_evaluation.ipynb** - Evaluate & analyze

## Using Python Modules Directly

```python
import pandas as pd
from src.data_loader import load_csv
from src.preprocessing import convert_columns_to_numeric
from src.feature_engineering import apply_log_transformation
from src.models import train_kmeans
from src.evaluation import anova_test

# Load and process data
df = load_csv('data/raw/merged_final.csv')
df = convert_columns_to_numeric(df, ['GOLD_PRICE', 'VIX'])
df = apply_log_transformation(df, ['VIX', 'GPR_INDEX'])

# Train model
from sklearn.preprocessing import StandardScaler
X = df[['VIX_log', 'GPR_INDEX_log', 'REAL_YIELD']].dropna()
X_scaled, scaler = StandardScaler().fit_transform(X), StandardScaler()
labels, model = train_kmeans(X_scaled, n_clusters=4)

# Evaluate
df['Regime'] = labels
results = anova_test(df, 'GOLD_CHANGE_%', 'Regime')
```

## Common Tasks

### Run Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/test_preprocessing.py  # Specific module
pytest -k test_name -v              # Specific test
```

### Format Code
```bash
black src/                          # Format all
black src/models.py                 # Specific file
```

### Check Code Style
```bash
flake8 src/ --max-line-length=100
```

### Generate Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

## Data Preparation

### Adding Your Data

1. Place CSV files in `data/raw/`
2. Load in notebook:
```python
df = load_csv('data/raw/your_file.csv')
```

3. Clean using preprocessing functions:
```python
from src.preprocessing import (
    convert_columns_to_numeric,
    standardize_date_column,
    drop_null_rows
)

df = standardize_date_column(df, 'DATE')
df = convert_columns_to_numeric(df, ['PRICE', 'VOLUME'])
df = drop_null_rows(df)
df.to_csv('data/processed/cleaned.csv', index=False)
```

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution:** Make sure you're in the project root directory:
```bash
cd geopolitical-risk-analysis
python -c "import sys; sys.path.insert(0, '.'); from src import models"
```

### Missing Data Errors
```
FileNotFoundError: data/raw/merged_final.csv not found
```
**Solution:** Check file path and ensure data is in `data/raw/` directory:
```bash
ls -la data/raw/  # List files in data directory
```

### Dependency Issues
```
ImportError: No module named pandas
```
**Solution:** Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Jupyter Issues
```
Kernel crashed or not responding
```
**Solution:** Restart kernel or reinstall:
```bash
pip install --upgrade notebook jupyter ipykernel
```

### Memory Issues with Large Datasets
**Solution:** Process in chunks:
```python
chunksize = 50000
for chunk in pd.read_csv('data/raw/huge_file.csv', chunksize=chunksize):
    # Process chunk
    df = process(chunk)
    df.to_csv('data/processed/output.csv', mode='a', header=False)
```

## Performance Tips

### Speed Up Analysis
```python
# Use numeric dtypes
df['numeric_col'] = pd.to_numeric(df['numeric_col'], downcast='float')

# Process subset for development
df_sample = df.sample(frac=0.1, random_state=42)

# Use pandas' built-in parallelization
import pandas as pd
pd.set_option('compute.use_numba', True)
```

### Reduce Memory Usage
```python
# Read only needed columns
df = pd.read_csv('file.csv', usecols=['DATE', 'PRICE', 'VOLUME'])

# Use categorical for text columns
df['Category'] = df['Category'].astype('category')

# Delete temporary variables
del df_temp
gc.collect()
```

## Getting Help

1. **Check documentation**: `docs/METHODOLOGY.md`
2. **Review examples**: Check `notebooks/` for examples
3. **Read docstrings**: `help(function_name)`
4. **Search issues**: GitHub Issues
5. **Ask questions**: GitHub Discussions

## Next Steps

- [ ] Review README.md
- [ ] Run 01_data_collection.ipynb
- [ ] Prepare your own data
- [ ] Customize clustering parameters
- [ ] Add your analysis notebooks
- [ ] Share results!

## Resources

- **Pandas**: https://pandas.pydata.org/docs/
- **scikit-learn**: https://scikit-learn.org/stable/
- **Jupyter**: https://jupyter.org/documentation
- **Matplotlib**: https://matplotlib.org/stable/contents.html
- **Seaborn**: https://seaborn.pydata.org/

---

**Happy analyzing!** 🎉

For issues or questions, open a GitHub Issue or Discussion.
