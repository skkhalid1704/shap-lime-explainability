# Explainability in Machine Learning Models Using SHAP and LIME

Code for the paper "Explainability in Machine Learning Models Using SHAP and LIME" by Khalid Ahamed Shaik, Golibjon Rakhmonov and Alexander I. Iliev (SRH University Berlin). Accepted at MEDES 2026 (Springer CCIS), to be published.

## What it does

Compares SHAP and LIME feature attributions for Logistic Regression, Random Forest and XGBoost on two tabular datasets:

- Telco Customer Churn (Kaggle: `blastchar/telco-customer-churn`)
- UCI Adult Income (OpenML `adult`, v2, 12,000-row stratified sample)

Both methods explain the same 100 test instances. The notebook measures:

- SHAP vs LIME agreement (Spearman rank correlation with bootstrap CI, top-10 overlap)
- LIME stability (Jaccard of top-10 features over 20 runs on 30 instances)
- SHAP vs native tree (Gini) importance
- Runtime per explanation
- Stability difference between datasets (Mann-Whitney U)

All settings (seed 42, sample sizes, LIME parameters) are in the configuration cell.

## Run

```bash
pip install kagglehub matplotlib numpy pandas scikit-learn scipy shap lime xgboost
jupyter notebook shap_lime_experiments.ipynb
```

Run the cells in order. Tables and figures are written to `outputs/`. The Telco download needs Kaggle access through `kagglehub`. The LIME stability loop is the slow part.
