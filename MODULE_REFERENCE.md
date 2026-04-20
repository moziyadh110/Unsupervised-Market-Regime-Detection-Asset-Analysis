# Module Reference Guide

## Quick Import Reference

```python
# Data Loading
from src.data_loader import load_csv, save_csv, display_basic_info

# Preprocessing
from src.preprocessing import (
    convert_columns_to_numeric,
    impute_missing_values,
    standardize_date_column,
    sort_by_date
)

# Feature Engineering
from src.feature_engineering import (
    apply_log_transformation,
    compute_rolling_volatility,
    compute_rolling_returns,
    compute_real_yield
)

# Models
from src.models import (
    train_kmeans,
    train_gmm,
    train_hmm,
    find_optimal_k,
    standardize_features
)

# Evaluation
from src.evaluation import (
    analyze_regime_profiles,
    anova_test,
    get_regime_statistics
)

# Visualization
from src.visualization import (
    plot_time_series,
    plot_correlation_heatmap,
    plot_boxplot_by_group
)
```

## Function Reference

### data_loader.py
- `load_csv(file_path)` - Load CSV file
- `save_csv(df, file_path)` - Save to CSV
- `display_basic_info(df)` - Show data overview
- `check_missing_values(df, column)` - Find NaNs

### preprocessing.py
- `convert_columns_to_numeric(df, columns)` - Type conversion
- `impute_missing_values(df, column, indices)` - Fill NaNs
- `remove_duplicates(df, subset, keep)` - Remove dupes
- `standardize_date_column(df, column, dayfirst)` - Parse dates
- `sort_by_date(df, column, ascending)` - Sort by date
- `drop_columns(df, columns)` - Remove columns
- `drop_null_rows(df, subset)` - Remove null rows

### feature_engineering.py
- `apply_log_transformation(df, columns)` - Log transform
- `compute_rolling_volatility(df, column, window)` - Volatility
- `compute_rolling_returns(df, column)` - % Returns
- `compute_real_yield(df, yield_col, inflation_col)` - Real yield
- `compute_rolling_average(df, column, window)` - Moving avg

### models.py
- `standardize_features(X)` - StandardScaler
- `find_optimal_k(X, k_range, metric)` - Find best k
- `train_kmeans(X, n_clusters)` - K-means
- `train_gmm(X, n_components)` - GMM
- `train_hmm(X, n_states)` - HMM
- `evaluate_clustering(labels, X)` - Metrics
- `assign_clusters_to_df(df, labels, column_name)` - Add labels

### evaluation.py
- `analyze_regime_profiles(df, regime_col, features)` - Mean by regime
- `analyze_target_by_regime(df, target, regime_col)` - Stats
- `anova_test(df, target, regime_col)` - F-test
- `get_regime_statistics(df, regime_col)` - Summary
- `regime_correlation_matrix(df, regime_label)` - Correlations

### visualization.py
- `plot_time_series(df, x_col, y_cols)` - Line plots
- `plot_correlation_heatmap(df)` - Heatmap
- `plot_distribution(df, column, bins)` - Histogram
- `plot_boxplot_by_group(df, x_col, y_col)` - Boxplot
- `plot_scatter_2d(df, x_col, y_col, color_col)` - Scatter
- `plot_cluster_sizes(labels)` - Cluster distribution
- `plot_regime_characteristics(df, regime_col)` - Regime profiles

## Example Usage

```python
# Complete workflow
import pandas as pd
from src import *

# 1. Load data
df = data_loader.load_csv('data/raw/data.csv')

# 2. Clean
df = preprocessing.convert_columns_to_numeric(df, ['PRICE', 'VOLUME'])
df = preprocessing.standardize_date_column(df, 'DATE')

# 3. Engineer features
df = feature_engineering.apply_log_transformation(df, ['VIX'])
df = feature_engineering.compute_real_yield(df)

# 4. Train model
X = df[['VIX_log', 'REAL_YIELD']].dropna()
X_scaled, scaler = models.standardize_features(X)
labels, model = models.train_kmeans(X_scaled, n_clusters=4)

# 5. Evaluate
df['Regime'] = labels
profiles = evaluation.analyze_regime_profiles(df, 'Regime')
results = evaluation.anova_test(df, 'GOLD_CHANGE_%', 'Regime')

# 6. Visualize
visualization.plot_boxplot_by_group(df, 'Regime', 'GOLD_CHANGE_%')
```
