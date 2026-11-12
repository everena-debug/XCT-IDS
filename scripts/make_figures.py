"""Figures for the IJIES version: no sub-figure labels inside images (labels are placed below in Word)."""
import glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5,
                     "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7})
import sys
X = (sys.argv[1] if len(sys.argv) > 1 else "xct_ids_v2_full").rstrip("/") + "/"
T = X + "tables/"
O = (sys.argv[2] if len(sys.argv) > 2 else "rebuilt_figures").rstrip("/") + "/"
import os; os.makedirs(O, exist_ok=True)
C = ['Normal', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
MIN = ['Analysis', 'Backdoor', 'DoS', 'Reconnaissance', 'Shellcode', 'Worms']
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"
tr = pd.read_csv(T + "results_transformer.csv"); xg = pd.read_csv(T + "results_xgboost.csv")


def save(fig, name):
    fig.tight_layout(); fig.savefig(O + name, dpi=300); plt.close(fig)


# realism: DCR ratio and coverage per class (seed 0, all generated records)
s = pd.read_csv(T + "synthetic_diagnostics.csv")
s0 = s[(s.seed == 0) & (s.set == "all")]
fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.4))
w = 0.26
for k, (g, col, lab) in enumerate([("ctgan", BLUE, "CTGAN"), ("ctganD", ORANGE, "CTGAN-D"), ("tvae", GREEN, "TVAE")]):
    d = s0[s0.generator == g].set_index("class").loc[MIN]
    ax[0].bar(np.arange(6) + (k - 1) * w, d.dcr_ratio, w, color=col, label=lab)
    ax[1].bar(np.arange(6) + (k - 1) * w, 100 * d.coverage_k5, w, color=col, label=lab)
for a in ax:
    a.set_xticks(range(6)); a.set_xticklabels(MIN, rotation=30, ha="right"); a.grid(axis="y", alpha=.3)
ax[0].axhline(1, ls="--", c="k", lw=.8); ax[0].set_ylabel("DCR ratio")
ax[1].set_ylabel("Coverage, k = 5 (%)"); ax[0].legend(frameon=False)
save(fig, "realism.png")

# discriminator SHAP importance (no title)
import json
imp = {}
for f in sorted(glob.glob(X + "tables/discriminator_shap_importance*.csv")):
    pass
try:
    di = pd.read_csv(T + "discriminator_shap_importance.csv")
    col = [c for c in di.columns if "shap" in c.lower() or "importance" in c.lower()][-1]
    fcol = [c for c in di.columns if "feature" in c.lower()][0]
    di = di.groupby(fcol)[col].mean().sort_values(ascending=False).head(15)[::-1]
    fig, a = plt.subplots(figsize=(6.0, 4.0))
    a.barh(di.index, di.values, color=BLUE); a.set_xlabel("Mean |TreeSHAP| of the real-vs-synthetic discriminator (log-odds)")
    a.grid(axis="x", alpha=.3); save(fig, "disc_shap.png")
except FileNotFoundError:
    print("no discriminator importance csv")

# main results: binary F1 and macro-F1 of all configurations
lab = {"real_only": "Transformer: real data only", "smote": "Transformer: SMOTE", "adasyn": "Transformer: ADASYN",
       "ctgan_nofilter": "Transformer: CTGAN, no filter", "ctgan_shap": "Transformer: CTGAN + SHAP refinement",
       "ctgan_random": "Transformer: CTGAN + random removal", "ctgan_conf": "Transformer: CTGAN + confidence filter",
       "ctgan_shaponly": "Transformer: CTGAN + SHAP-only score", "ctganD_nofilter": "Transformer: CTGAN-D, no filter",
       "ctganD_shap": "Transformer: CTGAN-D + SHAP refinement", "ctganD_random": "Transformer: CTGAN-D + random removal",
       "tvae_nofilter": "Transformer: TVAE, no filter", "tvae_random": "Transformer: TVAE + random removal",
       "tvae_shap": "XCT-IDS: TVAE + SHAP refinement"}
rows = []
for v, n in lab.items():
    d = tr[tr.variant == v]; rows.append((n, 100 * d.bin_f1_attack.mean(), 100 * d.bin_f1_attack.std(), 100 * d.multi_macro_f1.mean(), 100 * d.multi_macro_f1.std(), "t" if v != "tvae_shap" else "x"))
for v in xg.variant.unique():
    d = xg[xg.variant == v]
    n = "XGBoost: " + v.replace("xgb-", "").replace("real_only", "real data only").replace("_nofilter", ", no filter").replace("_shap", " + SHAP refinement").replace("_random", " + random removal").replace("ctganD", "CTGAN-D").replace("ctgan", "CTGAN").replace("tvae", "TVAE")
    rows.append((n, 100 * d.bin_f1_attack.mean(), 100 * d.bin_f1_attack.std(), 100 * d.multi_macro_f1.mean(), 100 * d.multi_macro_f1.std(), "g"))
R = pd.DataFrame(rows, columns=["n", "b", "bs", "m", "ms", "k"]).sort_values("m")
cmap = {"t": BLUE, "x": RED, "g": GREY}
fig, ax = plt.subplots(1, 2, figsize=(6.3, 4.6), sharey=True)
y = np.arange(len(R))
ax[0].barh(y, R.b, xerr=R.bs, color=[cmap[k] for k in R.k], ecolor="k", capsize=2)
ax[1].barh(y, R.m, xerr=R.ms, color=[cmap[k] for k in R.k], ecolor="k", capsize=2)
ax[0].set_yticks(y); ax[0].set_yticklabels(R.n, fontsize=6.3)
ax[0].set_xlim(85, 92); ax[1].set_xlim(43, 54.5)
ax[0].set_xlabel("Binary F1, attack class (%)"); ax[1].set_xlabel("Ten-class macro-F1 (%)")
for a in ax: a.grid(axis="x", alpha=.3)
save(fig, "main_results.png")

# confusion matrices
models = [("real_only", tr[tr.variant == "real_only"], "cm_real_only_s{}.csv"),
          ("tvae_shap", tr[tr.variant == "tvae_shap"], "cm_tvae_shap_s{}.csv"),
          ("xgb", xg[xg.variant == "xgb-real_only"], "cm_xgb-real_only_s{}.csv")]
fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.1))
for a, (_, df, _) in zip(ax, models):
    m = np.array([[df.bin_tn.mean(), df.bin_fp.mean()], [df.bin_fn.mean(), df.bin_tp.mean()]]); r = m / m.sum(1, keepdims=True)
    a.imshow(r, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{m[i, j]:,.0f}\n({100 * r[i, j]:.1f}%)", ha="center", va="center", fontsize=6.5, color="white" if r[i, j] > .5 else "black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["Normal", "Attack"]); a.set_yticklabels(["Normal", "Attack"])
    a.set_xlabel("Predicted"); a.set_ylabel("True")
save(fig, "cm_binary.png")
fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.6))
for a, (_, _, pat) in zip(ax, models):
    cm = sum(pd.read_csv(T + pat.format(s), index_col=0).values for s in range(5)); r = cm / cm.sum(1, keepdims=True)
    a.imshow(r, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if r[i, j] >= .005: a.text(j, i, f"{100 * r[i, j]:.0f}", ha="center", va="center", fontsize=4.2, color="white" if r[i, j] > .5 else "black")
    a.set_xticks(range(10)); a.set_yticks(range(10)); a.set_xticklabels(C, rotation=60, ha="right", fontsize=5); a.set_yticklabels(C, fontsize=5)
    a.set_xlabel("Predicted class")
ax[0].set_ylabel("True class")
save(fig, "cm_multi.png")

# ROC (seed 0)
yl = np.load(X + "preds/test_labels.npz"); yb, ym = yl["y_binary"], yl["y_multi"]
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
for v, n, c in [("tvae_shap", "XCT-IDS", RED), ("real_only", "Transformer, real only", BLUE), ("ctgan_shap", "CTGAN + SHAP", ORANGE),
                ("smote", "SMOTE", GREEN), ("xgb-real_only", "XGBoost", GREY)]:
    p = np.load(X + f"preds/{v}_s0.npz")["p_binary"]; f, t, _ = roc_curve(yb, p)
    ax[0].plot(f, t, c=c, lw=1, label=f"{n} (AUC = {auc(f, t):.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].legend(frameon=False, loc="lower right")
P = np.load(X + "preds/tvae_shap_s0.npz")["p_multi"]
for i, c in enumerate(C):
    f, t, _ = roc_curve(ym == i, P[:, i]); ax[1].plot(f, t, lw=1, label=f"{c} ({auc(f, t):.3f})")
ax[1].plot([0, 1], [0, 1], "k:", lw=.8); ax[1].legend(frameon=False, fontsize=5.5, loc="lower right")
for a in ax: a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate"); a.grid(alpha=.3)
save(fig, "roc.png")

# training curves (seed 0)
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.4))
for v, n, c in [("real_only", "Real data only", BLUE), ("ctgan_shap", "CTGAN + SHAP refinement", ORANGE), ("tvae_shap", "XCT-IDS (TVAE + SHAP refinement)", GREEN)]:
    h = pd.read_csv(X + f"logs/history_{v}_s0.csv")
    ax[0].plot(h.epoch, h.train_loss, "o-", ms=2.5, c=c, lw=1, label=n); ax[1].plot(h.epoch, h.val_score, "o-", ms=2.5, c=c, lw=1, label=n)
ax[0].set_ylabel("Training loss (BCE + CE)"); ax[1].set_ylabel("Validation score")
for a in ax: a.set_xlabel("Epoch"); a.grid(alpha=.3); a.legend(frameon=False)
save(fig, "training.png")

# sensitivity (validation, CTGAN + SHAP, seed 0)
sv = pd.read_csv(T + "sensitivity_validation.csv")
fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.0))
for a, col in zip(ax, ["val_multi_macro_f1", "val_bin_f1"]):
    p = sv.pivot_table(index="ratio", columns="removal", values=col) * 100
    a.imshow(p.values, cmap="Blues", vmin=p.values.min() - .5, vmax=p.values.max() + .2)
    for i in range(3):
        for j in range(3): a.text(j, i, f"{p.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    a.set_xticks(range(3)); a.set_xticklabels([f"{x:.1f}" for x in p.columns]); a.set_yticks(range(3)); a.set_yticklabels([f"{x:.1f}" for x in p.index])
    a.set_xlabel("Removal fraction"); a.set_ylabel("Augmentation ratio")
save(fig, "sensitivity.png")

# classifier SHAP importance (CTGAN + SHAP model, mean +/- sd over seeds)
ci = pd.read_csv(T + "classifier_shap_importance.csv"); ci = ci[ci.variant == "ctgan_shap"]
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
for a, o in zip(ax, ["binary_P_attack", "multi_predicted_class"]):
    g = ci[ci.output == o].groupby(["seed", "feature"]).mean_abs_shap.mean().unstack(0)
    m = g.mean(1).sort_values(ascending=False).head(15)[::-1]; sd = g.std(1)[m.index]
    a.barh(m.index, m.values, xerr=sd.values, color=BLUE, ecolor="k", capsize=2); a.set_xlabel("Mean |SHAP value|"); a.grid(axis="x", alpha=.3)
save(fig, "clf_shap.png")

# fidelity
fd = pd.read_csv(T + "classifier_shap_fidelity.csv"); fd = fd[fd.variant == "ctgan_shap"].groupby(["output", "k"])[["mean_drop_top_shap", "mean_drop_random"]].mean()
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.2))
for a, o in zip(ax, ["binary_P_attack", "multi_predicted_class"]):
    f = fd.loc[o]
    a.plot(f.index, f.mean_drop_top_shap, "o-", c=BLUE, label="Top-|SHAP| features replaced")
    a.plot(f.index, f.mean_drop_random, "s--", c=ORANGE, label="Random features replaced")
    a.set_xlabel("Number of features replaced (k)"); a.set_ylabel("Mean drop in model output"); a.grid(alpha=.3); a.legend(frameon=False)
save(fig, "fidelity.png")
print("done")

# class counts by stage
L = pd.read_csv(T + "ledger_counts.csv").set_index("stage")
stg = [("official_train", "Official training partition"), ("train_after_dedup", "After duplicate removal"),
       ("train_split", "Training split"), ("train_set_tvae_shap_seed0", "After augmentation and refinement")]
fig, a = plt.subplots(figsize=(6.3, 2.5)); w = 0.2
for k, (sname, lab) in enumerate(stg):
    a.bar(np.arange(10) + (k - 1.5) * w, L.loc[sname, C].astype(float), w, label=lab)
a.set_yscale("log"); a.set_xticks(range(10)); a.set_xticklabels(C, rotation=30, ha="right"); a.set_ylabel("Records (log scale)")
a.legend(frameon=False, ncol=2, fontsize=6); a.grid(axis="y", alpha=.3); save(fig, "class_counts.png")

# per-class F1 heatmap
pc = pd.read_csv(T + "paper_per_class_f1.csv").set_index("variant")
sel_ = [("tvae_shap", "XCT-IDS: TVAE + SHAP"), ("tvae_random", "TVAE + random removal"), ("tvae_nofilter", "TVAE, no filter"),
        ("ctgan_shap", "CTGAN + SHAP"), ("ctgan_random", "CTGAN + random removal"), ("ctgan_nofilter", "CTGAN, no filter"),
        ("ctganD_shap", "CTGAN-D + SHAP"), ("real_only", "Real data only"), ("smote", "SMOTE"), ("adasyn", "ADASYN"),
        ("xgb-real_only", "XGBoost: real data only"), ("xgb-tvae_shap", "XGBoost: TVAE + SHAP")]
M_ = np.array([[pc.loc[v, "F1_" + c] for c in C] for v, _ in sel_])
fig, a = plt.subplots(figsize=(6.3, 3.3))
a.imshow(M_, cmap="viridis", aspect="auto")
for i in range(M_.shape[0]):
    for j in range(10): a.text(j, i, f"{M_[i, j]:.1f}", ha="center", va="center", fontsize=5.5, color="black" if M_[i, j] > 60 else "white")
a.set_xticks(range(10)); a.set_xticklabels(C, rotation=30, ha="right"); a.set_yticks(range(len(sel_))); a.set_yticklabels([n for _, n in sel_], fontsize=6.3)
save(fig, "per_class.png")
