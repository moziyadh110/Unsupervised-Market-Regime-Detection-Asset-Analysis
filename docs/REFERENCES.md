# References & Data Sources

## Academic Papers

### Geopolitical Risk & Financial Markets

1. **Caldara, D., & Iacoviello, M. (2022)**
   - Title: "Measuring Geopolitical Risk"
   - Journal: International Finance Discussion Papers, Board of Governors of the Federal Reserve System
   - Link: https://www2.bc.edu/matteo-iacoviello/gpr.htm
   - Key Contribution: Development of the Geopolitical Risk (GPR) Index

2. **Iacoviello, M. (2016)**
   - Title: "Financial Business Cycles"
   - Journal: Review of Economic Dynamics
   - Focus: Links between financial conditions and macroeconomic cycles

3. **Baur, D. G., & Lucey, B. M. (2010)**
   - Title: "Is Gold a Hedge or a Safe Haven? An Analysis of Stocks, Bonds and Gold"
   - Journal: Financial Review, 45(2), 217-229
   - Key Finding: Gold acts as safe-haven asset during financial crises

### Precious Metals Analysis

4. **Tufano, P. (1998)**
   - Title: "The Economics of Precious Metal Markets"
   - Journal: Journal of Political Economy
   - Focus: Price dynamics and industrial demand

5. **Soytas, U., Sari, R., Hammoudeh, S., & Hacihasanoglu, E. (2009)**
   - Title: "World Oil Prices, Precious Metal Prices and Macroeconomy in Turkey"
   - Journal: Energy Policy, 37(12)
   - Focus: Commodity price correlations

### Clustering & Regime Identification

6. **Hamilton, J. D. (1989)**
   - Title: "A New Approach to the Economic Analysis of Nonstationary Time Series"
   - Journal: Econometrica, 57(2), 357-384
   - Key Method: Markov-switching models for regime identification

7. **Guidolin, M., & Timmermann, A. (2007)**
   - Title: "Asset Allocation Under Multivariate Regime Switching"
   - Journal: Journal of Economic Dynamics and Control, 31(11)
   - Focus: Using regime models for portfolio allocation

## Data Sources

### Primary Data Sources

#### Precious Metals Prices
- **Source**: Investing.com
- **Data Type**: Daily OHLC (Open, High, Low, Close)
- **Frequency**: Trading days only
- **Coverage**: Gold ($/oz), Silver ($/oz)
- **Period**: January 2000 - December 2024

#### Geopolitical Risk Index (GPR)
- **Source**: Matteo Iacoviello, Federal Reserve Board
- **URL**: https://www2.bc.edu/matteo-iacoviello/gpr.htm
- **Components**:
  - GPR_INDEX: Overall geopolitical risk measure
  - GPRD_ACTS: Realized geopolitical acts
  - GPRD_THREAT_LEVEL: Threat level indicator
- **Methodology**: Based on news articles mentioning geopolitical risk
- **Update Frequency**: Monthly

#### Economic Indicators
- **Federal Reserve Economic Data (FRED)**
  - URL: https://fred.stlouisfed.org/
  - Series Used:
    - DGS10: 10-Year Treasury Constant Maturity Rate
    - FEDFUNDS: Effective Federal Funds Rate
    - CPIAUCSL: Consumer Price Index, All Urban Consumers
    - DEXUSEU: USD/Euro Exchange Rate
  - Frequency: Daily/Monthly

#### Market Data
- **Yahoo Finance**
  - S&P 500 Index (^GSPC)
  - WTI Crude Oil (CL=F)
  - Frequency: Daily

- **CBOE (Chicago Board Options Exchange)**
  - VIX Index (Volatility Index)
  - URL: https://www.cboe.com/vix/
  - Description: 30-day implied volatility from S&P 500 options
  - Frequency: Daily

- **U.S. Federal Reserve**
  - US Dollar Index (Trade-Weighted)
  - Frequency: Daily

### Data Integration & Processing

All data was merged by date to create a unified dataset with:
- **9,291 observations** (trading days from 2000-2024)
- **22 features** across multiple dimensions
- **Time alignment**: Daily frequency with forward-fill for non-trading days when necessary

## Methodology References

### Machine Learning Methods

#### K-Means Clustering
- **Reference**: MacQueen, J. (1967). "Some Methods for classification and Analysis of Multivariate Observations"
- **Implementation**: scikit-learn, Lloyd's algorithm
- **Key Properties**: Partitional clustering, requires K specification

#### Gaussian Mixture Models (GMM)
- **Reference**: Fraley, C., & Raftery, A. E. (2002). "Model-Based Clustering, Discriminant Analysis, and Density Estimation"
- **Implementation**: Expectation-Maximization algorithm
- **Key Properties**: Probabilistic, soft clustering

#### Hidden Markov Models (HMM)
- **Reference**: Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition"
- **Implementation**: hmmlearn library
- **Key Properties**: Temporal modeling, state transitions

### Evaluation Metrics

#### Silhouette Score
- **Formula**: s(i) = (b(i) - a(i)) / max(a(i), b(i))
- **Range**: -1 to 1
- **Reference**: Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"

#### ANOVA (Analysis of Variance)
- **Purpose**: Test if group means differ significantly
- **Assumptions**: Normality, homogeneity of variance, independence
- **Alternative**: Kruskal-Wallis test (non-parametric)

## Software & Libraries

### Python Packages Used

```
pandas (1.3.0+)          # Data manipulation
numpy (1.21.0+)          # Numerical computing
scikit-learn (1.0.0+)    # Machine learning
scipy (1.7.0+)           # Scientific computing
matplotlib (3.4.0+)      # Visualization
seaborn (0.11.0+)        # Statistical visualization
jupyter (1.0.0+)         # Interactive notebooks
hmmlearn (0.2.7+)        # Hidden Markov Models
```

### Statistical Tools Used
- Pearson Correlation
- Z-score Standardization
- Moving Averages
- Log Transformation
- Rolling Standard Deviation

## Key Concepts

### Safe-Haven Assets
Assets that investors purchase during times of market uncertainty:
- Typically non-correlated with stocks
- Preserve value during crisis
- Gold and Treasury bonds are classic examples

### Geopolitical Risk
Defined as the risk associated with wars, terrorism, and political instability:
- Can impact commodity prices
- Affects investor risk appetite
- Often measured via news-based indices (like GPR)

### Market Regimes
Distinct periods characterized by different:
- Risk levels (VIX)
- Return patterns
- Correlation structures
- Volatility characteristics

### Real Yield
The return adjusted for inflation:
- Real Yield = Nominal Yield - Expected Inflation
- Key driver of commodity demand
- Negative real yields favor precious metals

## Data Quality Notes

### Missing Values
- **SILVER_PRICE**: 4 missing observations in original data (rows 1486, 7733, 7993, 8336)
- **Solution**: Imputed using rolling average of neighboring values
- **Justification**: Financial time series smoothness

### Outliers
- Detected using z-score method (|z| > 3)
- Mostly in VIX and GPR during crisis periods
- Retained as they represent real market events

### Data Limitations
- No corporate actions adjustment for indices
- Survivorship bias in equity indices
- GPR Index methodology changes over time
- Currency effects on USD-denominated precious metals

## Future Data Updates

### Recommended Update Schedule
- **Daily**: Precious metal prices, VIX, market indices
- **Monthly**: GPR Index, economic indicators (CPI, Treasury yields)
- **Quarterly**: Analysis updates and model retraining

### Data Validation Checks
Before running analysis, verify:
1. No gaps in date sequence
2. All mandatory columns present
3. Numeric columns contain only numbers (post-conversion)
4. Date ranges overlap appropriately
5. No unexpected duplicate records

## Contact & Attribution

For questions about data sources or methodology:
- **Geopolitical Risk Index**: Contact Matteo Iacoviello (matteo.iacoviello@bc.edu)
- **Federal Reserve Data**: Submit inquiry at https://fred.stlouisfed.org/
- **Project Data**: See README.md for project contact information

## Citation

If using this analysis, please cite:
```
Mohamed, M. Z. (2026). "Impact of Geopolitical Risks on Gold and Silver Prices 
Using Unsupervised Machine Learning." Data Mining & Analysis Project, 2026.
```

---

**Last Updated**: April 2026  
**Data Coverage**: January 2000 - December 2024  
**Total Records**: 9,291 daily observations
