# Generalized Random Forest (GRF) Analysis for Heterogeneous Treatment Effects

**Note**: This analysis uses proprietary experimental data from ongoing research and requires access to private preprocessing functions. The data files and preprocessing scripts are not included in this public repository for confidentiality reasons.

## Overview

This repository implements Generalized Random Forest (GRF) methods to estimate heterogeneous treatment effects in an experimental setting. The analysis focuses on identifying conditional average treatment effects (CATE) and local average treatment effects (CLATE) using machine learning-based causal inference methods.

## Repository Structure

```
├── GRF_func.R          # Core GRF analysis functions
├── main_GRF.R          # Main analysis script
└── README.md
```

## Key Features

### Implemented Methods

1. **Causal Forest**: For estimating heterogeneous treatment effects under randomized controlled trial (RCT) conditions
2. **Instrumental Forest**: For estimating local average treatment effects (LATE) with instrumental variables
3. **Feature Importance Analysis**: Using split frequencies and variable importance measures
4. **Policy Tree Construction**: For learning optimal treatment assignment rules
5. **Targeting Analysis**: Following Athey et al. (2025) methodology for targeted treatment assignment

### Treatment Comparisons

The analysis implements multiple treatment comparison frameworks:
- **T1 vs. Control**: Direct treatment effect estimation
- **T2 vs. Control (ITT)**: Intent-to-treat effects
- **T2 vs. Control (LATE)**: Local average treatment effects for compliers

### Machine Learning Integration

The codebase includes baseline machine learning methods for comparison:
- **Gradient Boosting Machine (GBM)**: Using `gbm` package
- **XGBoost**: For high-performance gradient boosting
- **Generalized Random Forest**: Baseline regression forest using `grf` package

## Core Functions

### Main Analysis (`GRF_func.R`)

- `run_GRF()`: Primary function for running GRF analysis across multiple time periods
- `analyze_feature_importance()`: Calculate variable importance using split frequencies
- `find_important_vars()`: Identify important variables across different models

### Policy Analysis (`GRF_func.R`)

- `targeting_analysis()`: Implement targeting policies for treatment assignment
- `analyze_tree_leaves()`: Analyze decision rules from policy trees
- `plot_policy_tree()`: Visualize learned treatment assignment policies

### Visualization (`GRF_func.R`)

- `plot_coef()`: Create coefficient plots with confidence intervals
- `plot_feature_importance()`: Visualize variable importance rankings
- `plot_split_pattern()`: Show split frequency patterns by tree depth
- `plot_ate_scatter()`: Scatter plots of average treatment effects


## References

1. Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of Statistics*, *47*(2), 1148-1178. https://doi.org/10.1214/18-AOS1709

2. Athey, S., Inoue, K., & Tsugawa, Y. (2025). Targeted Treatment Assignment Using Data from Randomized Experiments with Noncompliance. *AEA Papers and Proceedings*, *115*, 209–214. https://doi.org/10.1257/pandp.20251063

---

