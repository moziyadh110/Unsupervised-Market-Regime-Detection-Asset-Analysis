# Geopolitical Risk Impact on Gold & Silver Prices

A comprehensive data science project analyzing how geopolitical risks and hidden market regimes influence fluctuations in precious metal prices using unsupervised machine learning.

## 🎯 Hypothesis

**H(0):** There is no significant relationship (correlation) between geopolitical risk (GPR) and daily gold/silver price changes.

**H(1):** Geopolitical risks and market regimes significantly influence precious metal prices, which can be identified through clustering analysis.

## 📊 Project Overview

This project applies multiple machine learning clustering algorithms to identify distinct market regimes driven by geopolitical risks, treasury yields, inflation, and market volatility. The analysis supports the hypothesis that geopolitical events create identifiable market conditions affecting precious metals.

### Key Findings
- ✅ Identified 4 distinct market regimes: Calm/Bull, Financial Instability, Crisis/Panic, Inflationary Tightening
- ✅ K-means + temporal features achieved best performance (Silhouette Score: 0.2011)
- ✅ Gold price changes vary significantly across regimes (ANOVA p < 0.05)
- ✅ Successfully supports H(1): Geopolitical risk influences precious metal prices

## 📁 Project Structure

```
geopolitical-risk-analysis/
├── data/                          # Data files (raw and processed)
│   ├── raw/                       # Original datasets
│   └── processed/                 # Cleaned datasets
├── notebooks/                     # Jupyter notebooks (organized by analysis stage)
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_ml_clustering.ipynb
│   └── 06_model_evaluation.ipynb
├── src/                           # Reusable Python modules
│   ├── data_loader.py            # Load and validate data
│   ├── preprocessing.py          # Data cleaning functions
│   ├── feature_engineering.py    # Create derived features
│   ├── models.py                 # Clustering models (KMeans, GMM, HMM)
│   ├── evaluation.py             # Model evaluation metrics
│   └── visualization.py          # Plotting utilities
├── models/                        # Trained model files
│   └── kmeans_model.pkl
├── results/                       # Output plots and analysis
│   ├── plots/
│   └── analysis/
├── docs/                          # Documentation
│   ├── METHODOLOGY.md
│   └── REFERENCES.md
├── tests/                         # Unit tests
│   └── test_preprocessing.py
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── .gitignore                    # Git ignore rules
├── LICENSE                        # License file
├── Makefile                       # Common tasks
└── README.md                      # This file
```

## 🔍 Methodology

### Data Collection
- **Gold & Silver Prices:** Investing.com (daily OHLC data)
- **Geopolitical Risk Index (GPR):** GPR variants (overall, actual, threats)
- **Market Indicators:** VIX, Treasury yields, CPI, WTI crude oil, Fed funds rate, S&P 500, USD index
- **Time Period:** ~25 years, 9,291 daily observations
- **Features:** 22 variables covering geopolitical, economic, and market dimensions

### Data Cleaning
1. **Type Conversion:** Convert string columns to numeric
2. **Missing Value Imputation:** Forward/backward rolling average for 4 SILVER_PRICE NaNs
3. **Duplicate Removal:** Check for and remove duplicates
4. **Date Standardization:** Parse dates consistently
5. **Sorting:** Sort chronologically for time series analysis

### Feature Engineering
1. **Log Transformations:** Applied to skewed distributions (VIX, GPR, Fed funds rate)
2. **Volatility Measures:** 21-day rolling standard deviation for returns
3. **Returns Calculation:** Daily percentage changes for price series
4. **Real Yield:** Nominal yield - inflation rate
5. **Moving Averages:** 5-day rolling averages for smoothing

### Machine Learning Models

#### 1. **K-Means Clustering** (Primary Model)
- **Optimal k:** 4 clusters (determined by silhouette score)
- **Features:** VIX_log, GPR_INDEX_log, REAL_YIELD, USD_RET, SP500_RET
- **Result:** Silhouette Score = 0.2011

#### 2. **Gaussian Mixture Model** (GMM)
- **Components:** 4
- **Result:** Silhouette Score = 0.1859
- **Advantage:** Probabilistic cluster assignments

#### 3. **Hidden Markov Model** (HMM)
- **States:** 4
- **Approach:** Temporal sequence modeling
- **Result:** Captures state transitions over time

### Regime Identification
```
Regime 0: Calm / Bull Market
  - Low VIX, low GPR, positive real yields, strong USD
  - Positive gold returns expected

Regime 1: Financial Instability
  - Elevated VIX, moderate GPR, neutral yields
  - Volatile precious metal prices

Regime 2: Crisis / Panic
  - High VIX, high GPR, negative yields, weak USD
  - Flight-to-safety rally in gold/silver

Regime 3: Inflationary Tightening
  - Moderate VIX, persistent inflation, high rates
  - Gold under pressure from rising real yields
```

### Evaluation Metrics
- **Silhouette Score:** Measures cluster quality (-1 to 1, higher is better)
- **ANOVA Test:** Tests if gold returns differ significantly across regimes
- **Regime Profiles:** Mean values of key indicators per regime
- **Temporal Analysis:** How regimes evolve over time

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/geopolitical-risk-analysis.git
cd geopolitical-risk-analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Run Individual Notebooks
```bash
jupyter notebook notebooks/01_data_collection.ipynb
# Then work through notebooks 02-06 sequentially
```

#### Option 2: Use Python Modules Directly
```python
import pandas as pd
from src.data_loader import load_csv
from src.preprocessing import convert_columns_to_numeric, impute_missing_values
from src.feature_engineering import apply_log_transformation, compute_real_yield
from src.models import train_kmeans, evaluate_clustering
from src.evaluation import analyze_regime_profiles, anova_test

# Load data
df = load_csv('data/raw/merged_final_2.csv')

# Clean data
df = convert_columns_to_numeric(df, ['GOLD_PRICE', 'SILVER_PRICE', 'VIX'])

# Create features
df = apply_log_transformation(df, ['VIX', 'GPR_INDEX'])
df = compute_real_yield(df)

# Train model
from sklearn.preprocessing import StandardScaler
X = df[['VIX_log', 'GPR_INDEX_log', 'REAL_YIELD', 'USD_RET', 'SP500_RET']]
X_scaled, scaler = StandardScaler().fit_transform(X), StandardScaler()
labels, model = train_kmeans(X_scaled, n_clusters=4)

# Evaluate
eval_results = evaluate_clustering(labels, X_scaled)
profiles = analyze_regime_profiles(df, 'Regime')
anova_results = anova_test(df, 'GOLD_CHANGE_%', 'Regime')
```

#### Option 3: Use Makefile (Optional)
```bash
make install      # Install dependencies
make clean        # Remove cache files
make test         # Run tests
```

## 📚 Notebooks Overview

| Notebook | Purpose | Key Outputs |
|----------|---------|------------|
| `01_data_collection.ipynb` | Load and merge datasets | Merged CSV with all features |
| `02_data_cleaning.ipynb` | Handle missing values, type conversion | Clean dataset ready for analysis |
| `03_eda_analysis.ipynb` | Exploratory analysis, distributions, correlations | Summary statistics, visualizations |
| `04_feature_engineering.ipynb` | Log transforms, volatility, real yield | Feature-enriched dataset |
| `05_ml_clustering.ipynb` | Train 3 clustering models | Cluster assignments, model comparison |
| `06_model_evaluation.ipynb` | Statistical tests, regime profiling | ANOVA results, regime characteristics |

## 📈 Key Results

### Silhouette Scores by Model
```
K-Means (baseline)            : 0.1847
K-Means + Temporal Features   : 0.2011 ✅ BEST
Gaussian Mixture Model        : 0.1859
Hidden Markov Model           : ~0.16
```

### ANOVA Test Results
```
H(0): Gold returns are equal across regimes
Result: F-statistic = 24.3, p-value < 0.001
Conclusion: REJECT H(0) - Gold returns differ significantly by regime
```

### Regime Distribution
```
Regime 0 (Calm/Bull):       2,847 days (30.6%)
Regime 1 (Instability):     1,902 days (20.5%)
Regime 2 (Crisis/Panic):    1,823 days (19.6%)
Regime 3 (Inflation):       2,719 days (29.3%)
```

## 🔗 Data Sources

- **Gold & Silver:** [Investing.com](https://investing.com)
- **Geopolitical Risk Index (GPR):** [MATTEO IACOVIELLO'S WEBSITE](https://www2.bc.edu/matteo-iacoviello/gpr.htm)
- **Economic Indicators:** FRED Database, Federal Reserve
- **Market Data:** Yahoo Finance, CBOE (VIX)
- **Kaggle:**  https://www.kaggle.com/datasets/shreyanshdangi/gold-silver-price-vs-geopolitical-risk-19852025/data

## 📖 References

1. Caldara, D., & Iacoviello, M. (2022). "Measuring Geopolitical Risk." *International Finance Discussion Papers*, Board of Governors.
2. Iacoviello, M. (2016). "Financial Business Cycles." *Review of Economic Dynamics*.
3. Baur, D. G., & Lucey, B. M. (2010). "Is Gold a Hedge or a Safe Haven?" *Financial Review*, 45(2).
4. VIX: [Volatility Index](https://www.cboe.com/vix/)
5. scikit-learn: [Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✍️ Author

**Mohamed Ziyadh Mohamed**
- Project: Impact of Geopolitical Risks on Gold & Silver Prices
- Course: Introduction to Data Science (MSML-602)
- Institution: University of Maryland - College Park

## 🙏 Acknowledgments

- Data sources for providing comprehensive historical data
- scikit-learn, pandas, and matplotlib communities
- Course instructors and classmates for feedback

---

**Last Updated:** April 2026  
**Project Status:** ✅ Complete  
**Python Version:** 3.7+
