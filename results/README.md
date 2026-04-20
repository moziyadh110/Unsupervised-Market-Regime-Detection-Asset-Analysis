# Results Directory

Analysis outputs, plots, and findings are saved here.

## Structure

```
results/
├── plots/          # Generated visualizations
│   ├── correlation_heatmap.png
│   ├── time_series.png
│   ├── cluster_distribution.png
│   └── regime_boxplots.png
└── analysis/       # Analysis outputs
    ├── regime_profiles.csv
    ├── anova_results.txt
    └── silhouette_scores.csv
```

## Usage

Notebooks automatically save outputs here:
- Notebook 03 → EDA plots
- Notebook 05 → Clustering visualizations
- Notebook 06 → Statistical test results

## Note

Result files are excluded from Git to save space.
Re-generate by running the notebooks.
