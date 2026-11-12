"""Rebuild the main result tables of the paper from the released artifacts.

Usage:  python scripts/make_tables.py <extracted xct_ids_v2_full folder> [output folder]
Writes CSV files with means and standard deviations over five seeds and paired
95% confidence intervals for the principal comparisons.
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats

X = (sys.argv[1] if len(sys.argv) > 1 else "xct_ids_v2_full").rstrip("/") + "/"
O = (sys.argv[2] if len(sys.argv) > 2 else "rebuilt_tables").rstrip("/") + "/"
os.makedirs(O, exist_ok=True)
tr = pd.read_csv(X + "tables/results_transformer.csv")
xg = pd.read_csv(X + "tables/results_xgboost.csv")
allr = pd.concat([tr, xg], ignore_index=True)

BIN = ["bin_acc", "bin_precision", "bin_recall", "bin_specificity", "bin_f1_attack", "bin_mcc", "bin_roc_auc", "bin_pr_auc", "bin_gmean"]
MUL = ["multi_acc", "multi_macro_precision", "multi_macro_recall", "multi_macro_f1", "multi_weighted_f1", "multi_macro_auc", "novel_acc", "novel_macro_f1"]

def summary(cols, name):
    cols = [c for c in cols if c in allr.columns]
    g = allr.groupby("variant")[cols]
    out = (g.mean() * 100).round(2).astype(str) + " ± " + (g.std() * 100).round(2).astype(str)
    out.to_csv(O + name); print("wrote", O + name)

summary(BIN, "binary_results_mean_sd.csv")      # Table 8
summary(MUL, "multiclass_results_mean_sd.csv")  # Table 9

def paired(a, b, metric):
    x = allr[allr.variant == a].sort_values("seed")[metric].values * 100
    y = allr[allr.variant == b].sort_values("seed")[metric].values * 100
    d = x - y; n = len(d); h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return round(d.mean(), 2), round(d.mean() - h, 2), round(d.mean() + h, 2)

PAIRS = [("tvae_shap", "real_only"), ("tvae_shap", "tvae_random"), ("tvae_shap", "tvae_nofilter"), ("tvae_shap", "ctgan_shap"),
         ("tvae_shap", "smote"), ("tvae_shap", "adasyn"), ("tvae_shap", "xgb-real_only"), ("ctgan_shap", "real_only"),
         ("ctgan_shap", "ctgan_nofilter"), ("ctgan_shap", "ctgan_random"), ("ctgan_shap", "ctgan_conf"),
         ("ctgan_shap", "ctgan_shaponly"), ("ctganD_shap", "real_only")]
rows = []
for a, b in PAIRS:
    for m in ["multi_macro_f1", "bin_f1_attack"]:
        mean, lo, hi = paired(a, b, m)
        rows.append({"comparison": f"{a} - {b}", "metric": m, "mean_diff_pp": mean, "ci95_low": lo, "ci95_high": hi,
                     "significant": lo > 0 or hi < 0})
pd.DataFrame(rows).to_csv(O + "paired_differences.csv", index=False)  # Table 12
print("wrote", O + "paired_differences.csv")
