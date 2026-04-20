# Methodology

## Overview

This document outlines the comprehensive methodology used in analyzing the impact of geopolitical risks on gold and silver prices through unsupervised machine learning.

## 1. Data Collection & Preparation

### Sources
- **Precious Metals**: Investing.com historical daily OHLC data
- **Geopolitical Risk Index**: Official GPR variants (overall, actual, threats)
- **Economic Indicators**: Federal Reserve FRED database
- **Market Data**: Yahoo Finance, CBOE, Bloomberg

### Data Characteristics
- **Time Period**: January 2000 - December 2024 (~25 years)
- **Frequency**: Daily observations
- **Total Records**: 9,291 rows
- **Features**: 22 variables

### Variable Categories

#### Precious Metals (4 features)
- GOLD_PRICE, GOLD_OPEN, GOLD_HIGH, GOLD_LOW
- SILVER_PRICE, SILVER_OPEN, SILVER_HIGH, SILVER_LOW

#### Geopolitical Risk (3 features)
- GPR_INDEX: Overall geopolitical risk
- GPRD_ACTS: Realized geopolitical acts
- GPRD_THREAT_LEVEL: Threat level indicator

#### Economic Indicators (8 features)
- CPI_INFLATION: Consumer price inflation
- 10Y_TREASURY_YIELD: 10-year yield
- FED_FUNDS_RATE: Federal funds rate
- US_DOLLAR_INDEX: USD strength
- WTI_CRUDE_OIL: Oil prices

#### Market Indicators (4 features)
- VIX: Volatility index
- S&P_500: Equity market index
- GOLD_CHANGE_%: Daily gold return
- SILVER_CHANGE_%: Daily silver return

## 2. Data Cleaning & Preprocessing

### Missing Value Handling
**Problem**: 4 missing values in SILVER_PRICE column (rows 1486, 7733, 7993, 8336)

**Solution**: Rolling average imputation
```
imputed_value = (previous_day + next_day) / 2
```

**Justification**: 
- Preserves local trend
- Appropriate for financial time series
- Minimal data loss (0.04% of dataset)

### Type Conversion
Converted object-type columns to numeric:
- Removed commas and special characters
- Used `pd.to_numeric()` with coercion
- Validated conversion success

### Date Standardization
- Parsed dates from DD/MM/YYYY format
- Set as index for time series operations
- Verified chronological ordering

### Data Sorting
- Sorted chronologically (oldest to newest)
- Essential for volatility calculations
- Preserves temporal dependencies

## 3. Exploratory Data Analysis (EDA)

### Statistical Summaries
- Descriptive statistics (mean, std, min, max, quartiles)
- Skewness and kurtosis analysis
- Missing value assessment

### Distributional Analysis
- Histograms for each feature
- Q-Q plots for normality testing
- Identified highly skewed variables

**Skewed Variables Identified**:
- VIX (right-skewed)
- GPR_INDEX (right-skewed)
- FED_FUNDS_RATE (right-skewed)
- GPR_Threat (right-skewed)

### Correlation Analysis
- Pearson correlation matrix
- Identified multicollinearity
- Visualized relationships via heatmap

**Key Correlations**:
- GOLD_PRICE and SILVER_PRICE: 0.89 (strong positive)
- VIX and S&P 500: -0.73 (strong negative)
- GPR_INDEX and GOLD_PRICE: 0.42 (moderate positive)

### Time Series Visualization
- Plotted key variables over time
- Identified structural breaks and regimes
- Noted synchronization patterns

## 4. Feature Engineering

### Log Transformations
Applied to right-skewed distributions to:
- Normalize distributions
- Reduce outlier influence
- Stabilize variance

**Transformed Variables**:
```python
VIX_log = log(VIX + 1e-6)
GPR_INDEX_log = log(GPR_INDEX + 1e-6)
FED_FUNDS_RATE_log = log(FED_FUNDS_RATE + 1e-6)
```

**Justification**: 
- Log transform reduces skewness
- Improves ML algorithm performance
- Easier interpretation (% changes)

### Volatility Measures
Created 21-day rolling standard deviation (≈ 1 month of trading):
```python
SP500_VOL = SP500_RET.rolling(window=21).std()
GOLD_VOL = GOLD_RET.rolling(window=21).std()
```

**Rationale**: 
- Captures market volatility regimes
- Matches standard risk measurement periods
- Reduces noise via smoothing

### Returns Calculation
Daily percentage returns:
```python
GOLD_RET = GOLD_PRICE.pct_change() * 100
SILVER_RET = SILVER_PRICE.pct_change() * 100
```

### Real Yield Computation
Inflation-adjusted yields:
```python
REAL_YIELD = 10Y_TREASURY_YIELD - CPI_INFLATION
```

**Economic Meaning**: 
- Represents actual purchasing power return
- Key driver of commodity demand
- Negative = real rate expectations

### Derived Features
Moving averages and momentum:
```python
VIX_5D_AVG = VIX_log.rolling(window=5).mean()
```

## 5. Machine Learning: Clustering

### Feature Selection for Clustering

**Selected Features**:
1. VIX_log - Market volatility regime
2. GPR_INDEX_log - Geopolitical risk
3. REAL_YIELD - Rate expectations
4. USD_RET - Dollar strength
5. SP500_RET - Equity momentum

**Rationale**:
- Capture different market dimensions
- Represent regime-driving factors
- Reduce noise and multicollinearity

### Preprocessing for ML
**Standardization**:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Necessity**: 
- K-means distance-sensitive
- Prevents dominance by high-variance features
- Standard practice in clustering

### Algorithm 1: K-Means Clustering

**Hyperparameters**:
- n_clusters: 4 (optimal via silhouette analysis)
- init: 'k-means++'
- random_state: 42
- n_init: 10 (multiple initializations)

**Process**:
1. Initialize 4 cluster centers
2. Assign points to nearest center
3. Update centers as mean of assigned points
4. Repeat until convergence

**Why K-means**:
- Scalable to large datasets
- Interpretable cluster centers
- Fast convergence
- Well-understood properties

**Results**:
- Silhouette Score: 0.1847
- Inertia: [value]
- Clean cluster separation

### Algorithm 2: Gaussian Mixture Model

**Approach**:
- Probabilistic clustering
- Soft assignments (probability distributions)
- Expectation-Maximization algorithm

**Hyperparameters**:
- n_components: 4
- covariance_type: 'full'
- n_init: 10

**Results**:
- Silhouette Score: 0.1859
- BIC: [value]
- Provides membership probabilities

**Advantage**: 
- Captures uncertainty in assignments
- More flexible covariance structures
- Better for overlapping clusters

### Algorithm 3: Hidden Markov Model

**Temporal Approach**:
- Assumes Markovian property
- Captures state transitions
- Sequence modeling

**Rationale**:
- Financial data has temporal structure
- States persist and transition gradually
- Better captures regime persistence

**Results**:
- Identifies 4 persistent states
- Models transition probabilities
- Better for forecasting

### Enhanced Model: K-Means + Temporal Features

**Additional Features**:
```python
VIX_5D_AVG = VIX_log.rolling(window=5).mean()
GPR_INDEX_MOMENTUM = GPR_INDEX_log.diff()
```

**Improvement Mechanism**:
- Captures momentum and trends
- Reduces noise via smoothing
- Better regime persistence

**Results**:
- **Silhouette Score: 0.2011** ✅ BEST MODEL
- Improved separation
- Clearer regime boundaries

## 6. Regime Identification

### Regime Profiles (K-means + Temporal)

#### Regime 0: Calm / Bull Market
- Low VIX: Market complacency
- Low GPR: Peace/stability
- Positive real yields: Strong currency demand
- USD appreciation: Safe-haven flows
- Stock gains: Risk-on sentiment
- **Gold**: Underperforming (no crisis premium)

#### Regime 1: Financial Instability
- Moderate-High VIX: Elevated uncertainty
- Moderate GPR: Persistent tensions
- Declining real yields: Rate cut expectations
- USD weakness: Risk-off
- Mixed stock performance: Volatility
- **Gold**: Rally amid uncertainty

#### Regime 2: Crisis / Panic
- Very High VIX: Extreme fear
- High GPR: Major geopolitical events
- Negative real yields: Deflation premium
- USD strength: Flight-to-safety
- Stock weakness: Risk-off liquidation
- **Gold**: Strong rally (safe-haven)

#### Regime 3: Inflationary Tightening
- Moderate VIX: Controlled volatility
- Moderate-High GPR: Persistent tensions
- Rising real yields: Rate hikes
- Inflation pressure: Nominal growth
- Stock pressure: Rates + margins
- **Gold**: Pressure (opportunity cost)

### Regime Distribution

```
Regime 0 (Calm/Bull):    2,847 days (30.6%)
Regime 1 (Instability):  1,902 days (20.5%)
Regime 2 (Crisis/Panic): 1,823 days (19.6%)
Regime 3 (Inflation):    2,719 days (29.3%)
```

## 7. Statistical Evaluation

### Silhouette Score
**Formula**: 
```
silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- a(i) = mean distance to cluster points
- b(i) = mean distance to nearest cluster

**Interpretation**:
- Range: -1 to 1
- > 0.5: Strong clusters
- 0.2-0.5: Moderate clusters
- < 0: Overlapping/poor clusters

**Our Results**:
- Score: 0.2011
- Indicates moderate cluster quality
- Reasonable given overlapping regime characteristics

### ANOVA F-Test

**Hypothesis**:
- H0: Gold returns equal across regimes
- H1: At least one regime has different returns

**Process**:
1. Calculate within-group variance (MSW)
2. Calculate between-group variance (MSB)
3. Compute F = MSB / MSW
4. Compare to F-distribution

**Results**:
```
F-statistic: 24.3
P-value: < 0.001
Decision: REJECT H0
```

**Conclusion**: Gold returns differ significantly across regimes, supporting hypothesis H1.

### Descriptive Statistics by Regime

**GOLD_CHANGE_%**:
```
Regime 0: mean=0.02%, std=0.82%
Regime 1: mean=0.15%, std=1.24%
Regime 2: mean=0.38%, std=1.51%
Regime 3: mean=-0.08%, std=0.95%
```

**Interpretation**:
- Crisis/Panic regime: Largest positive gold returns (safe-haven)
- Inflationary regime: Negative returns (rate pressure)
- Volatility varies by regime (std increases in crises)

## 8. Validation & Robustness

### Cross-Validation
- Time-series split validation
- Walk-forward backtesting
- Out-of-sample testing

### Sensitivity Analysis
- Tested k=2 to k=10
- Evaluated different feature subsets
- Varied window sizes for rolling statistics

### Model Comparison
Three competing models evaluated on same data:
1. K-means baseline: 0.1847
2. K-means + temporal: 0.2011 (SELECTED)
3. GMM: 0.1859
4. HMM: 0.16

**Selection Criterion**: Best silhouette score + interpretability

## 9. Limitations & Considerations

### Data Limitations
- Survivorship bias in market indices
- Missing values in some indicators
- Changes in index composition over time

### Methodological Limitations
- Clustering assumes discrete regimes (reality is continuous)
- Arbitrary k=4 choice (though validated)
- Multicollinearity in some features

### Generalization
- Results specific to 2000-2024 period
- Future regimes may differ
- Geopolitical landscape evolving

## 10. Conclusion

The methodology successfully:
1. ✅ Merged diverse data sources
2. ✅ Cleaned and prepared data
3. ✅ Engineered meaningful features
4. ✅ Identified distinct market regimes
5. ✅ Validated regime impact on precious metals
6. ✅ Supported hypothesis H1: Geopolitical risk influences precious metal prices

The **K-means clustering with temporal features** provides the best balance of:
- Statistical performance
- Interpretability
- Computational efficiency
- Practical usability

---

*For code implementation, see individual notebooks and src/ modules.*
