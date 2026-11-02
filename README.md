# Freight Rate Prediction

Machine-learning take-home for the Spotter ML Engineer role: predict the posted price of a truckload.

## Business problem
A freight broker posts a price (`posted_rate`) for every load: pick-up city, delivery city, distance, trailer type, weight and date.
Given 48,000 priced loads from **January-October 2025**, predict the price of 12,000 unpriced loads from **November-December 2025**, and forecast one fixed lane
(Lexington to Fort Wayne, 360 mi, Dry Van, 32,000 lb) for each day of December.
Because the target loads lie in the **future**, this is a forecasting problem, and 8 of the 72 validation cities were never seen in training.

## Data (not committed; see "Data location")
| File | Content |
|---|---|
| `train_test.csv` | 48,000 labelled loads, 2025-01-01..2025-10-31 |
| `validation.csv` | 12,000 loads, 2025-11-01..2025-12-31, no label |
| `validation_predictions_template.csv` | the 12,000 `load_id`s to fill in |
| `december_chart_inputs.csv` | 31 rows for the fixed lane, one per December day |

## Model architecture
```
log(rate) = Linear[ log distance, log distance^2, equipment, daily market index, trend, quarter-end ramp ]
          + GradientBoosting[ log distance, equipment, |weight|, pickup/delivery lat-lon ]   (fitted on the linear model's residuals)
```
* Linear part: effects that must **extrapolate in time**. Trees cannot extrapolate.
* Tree part: non-linear weight and geography effects; coordinates let it handle cities it has never seen.
* ~1.4% of training labels are corrupted (off by roughly 2-6x) and are removed from training only.
* `quote_signal` is deliberately excluded (unstable proxy); the measured cost is about 0.04 MAPE points.

## Validation methodology
Three **forward-in-time** expanding-window folds (train Jan-Jun/test Jul-Aug, Jan-Jul/Aug-Sep, Jan-Aug/Sep-Oct), each with 8 cities removed from training to simulate unseen cities.
Metrics: MAE, MAPE, RMSE, on **trusted labels** and on **all labels** (corrupted rows included).

| | MAE | MAPE | RMSE |
|---|---|---|---|
| Final model, trusted labels | $41.6 | 1.77% | $59.9 |
| Final model, all labels | $94.5 | 4.1% | ~$620 |

These are cross-validation estimates, **not** validation-set scores (those labels are hidden). See `docs/METHODOLOGY.md` for assumptions, verified/unverified claims and known limitations.

## Project structure
```
freight-rate-challenge/
├── common.py                 cleaning, features, FreightRateModel (single source of truth)
├── eda.py                    command-line EDA (prints findings, saves figures)
├── train_and_predict.py      cross-validation, final fit, predictions
├── score.py                  provided scorer (unchanged)
├── requirements.txt
├── notebooks/                01_exploratory_data_analysis, 02_feature_engineering,
│                             03_model_development_and_validation, 04_final_predictions_and_submission
├── outputs/                  validation_predictions.csv, december_chart_inputs.csv, cv_results*.csv, logs
├── figures/                  EDA / report figures (figures/notebooks/ for the notebooks)
├── scorer_results/           candidate_december.png from score.py
├── validation_predictions.csv  final submission file (copy of outputs/validation_predictions.csv)
└── report.docx
```

## Installation (Python 3.10+)
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data location
Put the four CSVs in `data/` using the names above. The notebooks are more forgiving: they also find the files one folder **above** the project, and with hyphens
(`train-test.csv`), and **copy** them into `data/` (originals are never modified). The scripts expect `data/` with underscore names.

## Execution order
### Scripts
```bash
python eda.py                       # 1. explore  -> figures/
python train_and_predict.py         # 2. CV + train + predict -> outputs/   (~1 min; add --skip-cv to skip validation)
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
python make_report.py               # 3. report.docx
```
### Notebooks (run in order; each runs top to bottom, about 4 min in total, 03 is the slowest)
```bash
jupyter lab notebooks/                                  # interactive
# or headless, from the project root:
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 notebooks/01_exploratory_data_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 notebooks/03_model_development_and_validation.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 notebooks/04_final_predictions_and_submission.ipynb
```
The notebooks import `common.py` and `train_and_predict.py` and can be started from any folder (they search upwards for `common.py`).
Outputs go to `outputs/notebooks/` and `figures/notebooks/`; notebook 04 checks that its predictions match the script pipeline.

| Notebook | Purpose |
|---|---|
| 01 | eight EDA questions: files, quality, unseen cities, target, corrupted labels, market signals, time structure, December inputs |
| 02 | every feature: meaning, availability at prediction time, leakage, generalisation; feature-pipeline checks |
| 03 | baselines, official CV reproduced, leakage experiments (with positive control), ablations, placebo tests, residuals, metric checks |
| 04 | final fit, 12,000 predictions, submission file, December chart and sensitivity, `score.py` |

## Output files
| File | Description |
|---|---|
| `validation_predictions.csv` | final predictions: exactly `load_id,predicted_rate`, 12,000 rows, template order |
| `outputs/december_chart_inputs.csv` | the 31 December rows with `predicted_rate` filled in |
| `scorer_results/candidate_december.png` | chart produced by the provided `score.py` |
| `outputs/cv_results.csv`, `cv_results_by_fold.csv` | validation metrics |
| `report.docx` | validation approach and the December chart |

Loom walkthrough: _<paste link here>_

## Known caveats (details in `docs/METHODOLOGY.md`)
* The December curve relies on two assumptions that cannot be tested: the quarter-end premium seen in March, June and September also occurs in December, and the slow drift continues (sensitivity: about -2% on average / -4% on Dec 31 without the ramp).
* If the scorer includes corrupted validation labels, expect MAE around $94 and RMSE around $620 regardless of model quality.
* The daily market index is computed from the features of the rows being predicted; this is allowed here but would need a forecast in a live system.
* Results are reproducible in one environment; across library versions predictions can differ slightly (mean ~0.08%) because of scikit-learn's default early-stopping split.
