# XCT-IDS

Code and experimental artifacts for the paper:

**XCT-IDS: An Explainable Transformer Framework with Realism-Audited CTGAN and TVAE Augmentation and SHAP Analysis for Intrusion Detection and Attack Classification**
Gudipudi Balaiah, Prajna Bodapati, Velumani R. Submitted to the *International Journal of Intelligent Engineering and Systems* (IJIES).

The framework enlarges the minority attack classes of UNSW-NB15 with class-conditional tabular generators (CTGAN, a domain-constrained CTGAN-D and TVAE), audits the realism of the synthetic records, refines them with a cross-fitted XGBoost discriminator and TreeSHAP, and trains a multi-task FT-Transformer for binary detection and ten-class attack classification. Every configuration is run with five seeds and compared with matched filtering controls and with XGBoost, SMOTE and ADASYN baselines under the same protocol.

## Repository contents

| Item | Location |
|---|---|
| Experiment notebook (all stages) | `xct_ids_v2_full.ipynb` |
| Script that rebuilds the result tables | `scripts/make_tables.py` |
| Script that rebuilds the figures | `scripts/make_figures.py` |
| Python dependencies | `requirements.txt` |
| Full experimental outputs (347 MB) | Release **v1.0**, asset `xct_ids_v2_full.zip` |

### Contents of `xct_ids_v2_full.zip`

| Folder / file | What it contains | Reviewer item |
|---|---|---|
| `run_config.json` | Every setting of the run: seeds, split seed, minority classes, ratios, generator, discriminator, Transformer and XGBoost hyperparameters, feature lists, software versions | Configuration, seeds |
| `tables/split_train_rows.csv`, `tables/split_val_rows.csv` | Row indices (in the official training file) of the 86,192 training and 21,548 validation records after de-duplication | Split indices |
| `tables/data_audit.json`, `tables/ledger_counts.csv` | Duplicate counts, label conflicts, test overlap, per-class record ledger | Training ledger |
| `tables/preprocessing.json` | Numerical and categorical feature lists, categorical mappings and preprocessing statistics | Preprocessing mappings |
| `tables/generator_column_plan_*.json`, `tables/augmentation_policy.csv` | Column handling of each generator and class, augmentation targets | Generator configuration |
| `tables/retention_masks_{ctgan,ctganD,tvae}_seed{0-4}.csv.gz` | For every synthetic record: class, discriminator margin, synthetic probability, SHAP dominance score, quality score and the keep/remove decision of the SHAP rule, SHAP-only score, confidence filter and random removal | Per-record quality scores and retention masks |
| `tables/refinement_counts_*.csv` | Generated, removed and kept counts per class | Removal counts |
| `tables/synthetic_diagnostics.csv`, `realism_statistics.csv`, `discriminator_auc.csv`, `generator_domain_violations.csv` | Realism audit before and after filtering | Realism audit |
| `tables/results_transformer.csv`, `tables/results_xgboost.csv` | All metrics of every configuration and seed | Results |
| `tables/cm_*.csv` | Ten-class confusion matrices of every configuration and seed | Confusion matrices |
| `tables/paper_*.csv` | Paired differences, McNemar tests, per-class F1 | Statistics |
| `tables/sensitivity_validation.csv` | Sensitivity to the augmentation ratio and removal fraction (validation data) | Sensitivity |
| `tables/classifier_shap_*.csv` | Classifier SHAP importance, deletion fidelity and cross-seed stability | Explanations |
| `preds/*.npz`, `preds/test_labels.npz` | Test-set probabilities of every configuration and seed, and the test labels | Labels and predictions |
| `models/*.pt` | Trained FT-Transformer weights of every configuration and seed | Model weights |
| `logs/` | Training histories and run times | Training logs |
| `figures/` | Figures produced by the notebook | Figures |

## Data

UNSW-NB15 is not redistributed here. Download the training and testing CSV files (`UNSW_NB15_training-set.csv`, 175,341 records, and `UNSW_NB15_testing-set.csv`, 82,332 records) from the official dataset page of UNSW Canberra and set the data path in the first cell of the notebook. The notebook identifies the larger file as the official training partition, which resolves the swapped file names in some mirrors.

## Environment

The reported results were produced on an NVIDIA Tesla T4 (16 GB) with Python 3.13, PyTorch 2.11.0 (CUDA 13.0), XGBoost 3.4.1, ctgan 0.13.0, pandas 2.3.3 and NumPy 2.1.3. The complete five-seed study took 5.2 hours.

```bash
pip install -r requirements.txt
```

## Reproducing the results

**Rebuild the tables and figures from the released outputs (minutes, no GPU needed)**

1. Download `xct_ids_v2_full.zip` from Release v1.0 and extract it, for example to `xct_ids_v2_full/`.
2. Run:

```bash
python scripts/make_tables.py  xct_ids_v2_full  rebuilt_tables
python scripts/make_figures.py xct_ids_v2_full  rebuilt_figures
```

`make_tables.py` writes the binary and ten-class results (mean ± SD over five seeds) and the paired differences with 95% confidence intervals. `make_figures.py` writes the realism, confusion-matrix, ablation, sensitivity and explanation figures.

**Re-run the full experiment (GPU)**

Open `xct_ids_v2_full.ipynb`, set the data path and run all cells. The notebook de-duplicates the data, creates the stratified split (seed 42), trains the generators, refines the synthetic data, trains all Transformer and XGBoost configurations for seeds 0–4, runs the realism audit, statistics and explanations, and writes the same folder structure as the release.

## Protocol summary

- 67,601 exact duplicates (38.6%) removed from the official training partition. The remaining 107,740 records are split into 86,192 training and 21,548 validation records (stratified, seed 42).
- Generators, discriminators, preprocessing and classifiers are fitted on the training split only. The validation split is used for generator selection, early stopping and the sensitivity analysis.
- The official test partition (82,332 records) is used once, for final reporting, together with a de-overlapped novel subset of 73,791 records.
- Six minority classes (Analysis, Backdoor, DoS, Reconnaissance, Shellcode, Worms) are raised to 40% of the Normal count. Exactly 10% of the generated records of each class are removed by the refinement rule or by the matched control.

## Citation

If you use this code, please cite the paper (citation details will be added after publication).

## License

MIT License. See `LICENSE`.
