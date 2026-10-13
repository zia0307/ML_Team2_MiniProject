# Reproduce & Extend: Quantitative Trading with Machine Learning

UE24CS352A Machine Learning - Mini-Project. Team: Zia (PES2UG24CS620), Harshavardhan (PES2UG25CS817).

Reference: R. Glawion, *Quantitative Trading with Machine Learning* (Stanford CS229). Data and the author's saved predictions come from https://github.com/rglawion/cs229_project_report.

- **Part 1:** reproduces Elastic Net, Decision Tree, XGBoost, LightGBM (RMSE and trading profit, paper vs ours).
- **Part 2:** adds one new model, **RBF Support Vector Regression (SVR)**, on the same data, split and metrics.

## Files
| Path | What it is |
|---|---|
| `mini_project.ipynb` | **All the code**: data, rolling windows, 5 models, tables, trading simulation, figures |
| `data/` | processed price/feature data and the author's saved predictions (inputs) |
| `results/` | our saved test predictions (15 small files, one per model and horizon) |
| `docs/` | write-up PDF and slide deck |

## Run
```bash
pip install -r requirements.txt
jupyter notebook mini_project.ipynb      # then Run All (about 1 minute)
```
Run from the repository root (`data/` and `results/` must sit next to the notebook). **Google Colab:** upload the whole folder, or clone the repo, and Run All.

By default the notebook loads the saved predictions in `results/`. To retrain every model from scratch, set `RUN_FULL = True` in the first code cell (several hours on a single CPU). Section 3 of the notebook is a one-window live demo (about 1 minute).

## Main results (test RMSE, mean of monthly RMSEs; paper -> ours)
| Model | h=1 | h=5 | h=20 |
|---|---|---|---|
| Elastic Net | .0174 -> .0174 | .0401 -> .0401 | .0751 -> .0754 |
| Decision Tree | .0190 -> .0175 | .0422 -> .0407 | .0952 -> .0824 |
| XGBoost | .0176 -> .0179 | .0410 -> .0420 | .0850 -> .0857 |
| LightGBM | .0177 -> .0179 | .0408 -> .0406 | .0815 -> .0830 |
| **SVR (new)** | - -> .0183 | - -> .0417 | - -> .0774 |

## Deviations from the paper
- One CPU instead of the author's 64 cores / ~36 h: smaller hyper-parameter grids; XGBoost and LightGBM re-tuned every 12 / 6 months (refit monthly). Timings are not comparable.
- Only h = 1, 5, 20 (the poster's horizons). One random seed.
- Decision Tree grid uses `max_features` in {sqrt, log2}, which is probably why our trees beat the paper's (not tested).
- Trading profits use chronological order of predictions; the author's code orders each month's predictions by an unstable sort. The notebook checks that, with his predictions and ordering, our simulator reproduces all 96 cells of the paper's Table 2.
