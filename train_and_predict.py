"""
train_and_predict.py - validate, train, predict.

Run:  python train_and_predict.py

Steps
  1. Build the forward-in-time cross-validation and compare the final model with simpler alternatives.
  2. Train the final model on ALL of train_test.csv (after removing corrupted labels).
  3. Predict the 12,000 validation loads  -> outputs/validation_predictions.csv
  4. Predict the 31 December chart rows   -> outputs/december_chart_inputs.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from common import (COORD_COLS, FreightRateModel, clean_inputs, daily_market_index,
                    flag_label_outliers, load_data)

# Validation folds: (last training month, last test month).  Each test window is ~2 months long,
# exactly like the real task (train Jan-Oct -> predict Nov-Dec).
FOLDS = [(6, 8), (7, 9), (8, 10)]
N_HELD_OUT_CITIES = 8      # the real validation set has 8 cities never seen in training


# ---------------------------------------------------------------------------
# a deliberately naive baseline: "throw all raw columns into a GBM"
# ---------------------------------------------------------------------------
class NaiveGBM:
    """No cleaning, no calendar features, uses the raw market_index and quote_signal columns."""

    def fit(self, df, drop_outliers=False):
        df = clean_inputs(df)
        self.m = HistGradientBoostingRegressor(random_state=0).fit(self._x(df), np.log(df.posted_rate))
        return self

    def predict(self, df):
        return np.exp(self.m.predict(self._x(clean_inputs(df))))

    @staticmethod
    def _x(df):
        x = df[["distance", "weight", "market_index", "quote_signal", *COORD_COLS]].copy()
        x["equipment"] = df.equipment.astype("category").cat.codes
        return x


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(y, p) -> dict:
    return dict(MAE=np.mean(np.abs(p - y)),
                MAPE=100 * np.mean(np.abs(p - y) / y),
                RMSE=np.sqrt(np.mean((p - y) ** 2)))


def cross_validate(train: pd.DataFrame, dmi: pd.Series, out_dir: Path) -> pd.DataFrame:
    """Expanding-window, forward-in-time validation with unseen-city simulation."""
    # name -> (how to build the model, whether to drop corrupted training labels)
    candidates = {
        "1. Distance + equipment only (linear)":           (lambda: FreightRateModel(dmi, False, False, False, use_gbm=False), True),
        "2. Naive GBM on raw columns, no cleaning":        (lambda: NaiveGBM(), False),
        "3. Final model WITHOUT trend + quarter-end terms": (lambda: FreightRateModel(dmi, False, False, True), True),
        "4. Final model, labels NOT cleaned":              (lambda: FreightRateModel(dmi), False),
        "5. FINAL MODEL":                                  (lambda: FreightRateModel(dmi), True),
    }
    rng = np.random.RandomState(0)
    all_cities = sorted(set(train.pickup) | set(train.delivery))
    rows = []

    for fold, (last_train_month, last_test_month) in enumerate(FOLDS, 1):
        fit_part = train[train.date.dt.month <= last_train_month]
        test_part = clean_inputs(train[(train.date.dt.month > last_train_month) & (train.date.dt.month <= last_test_month)])

        # simulate the 8 unseen validation cities: remove every training row that touches them
        held_out = set(rng.choice(all_cities, N_HELD_OUT_CITIES, replace=False))
        touches = fit_part.pickup.isin(held_out) | fit_part.delivery.isin(held_out)
        fit_part = fit_part[~touches]
        test_unseen = (test_part.pickup.isin(held_out) | test_part.delivery.isin(held_out)).to_numpy()

        # which TEST labels are trustworthy?  (only used to REPORT; corrupted labels cannot be predicted by anyone)
        trusted = flag_label_outliers(test_part, dmi)
        y = test_part.posted_rate.to_numpy()
        print(f"fold {fold}: train {fit_part.date.min().date()}..{fit_part.date.max().date()} ({len(fit_part):,} rows)"
              f" -> test {test_part.date.min().date()}..{test_part.date.max().date()} ({len(test_part):,} rows,"
              f" {test_unseen.mean():.0%} touch unseen cities)")

        for name, (build, drop_outliers) in candidates.items():
            model = build().fit(fit_part, drop_outliers=drop_outliers)
            p = model.predict(test_part)
            for group, mask in [("all rows", np.ones(len(y), bool)), ("unseen cities", test_unseen)]:
                m = metrics(y[trusted & mask], p[trusted & mask])
                m_raw = metrics(y[mask], p[mask])
                rows.append(dict(model=name, fold=fold, group=group, **m, RMSE_incl_corrupted=m_raw["RMSE"]))

    res = pd.DataFrame(rows)
    summary = res.groupby(["model", "group"])[["MAE", "MAPE", "RMSE", "RMSE_incl_corrupted"]].mean().round(2)
    summary.to_csv(out_dir / "cv_results.csv")
    res.to_csv(out_dir / "cv_results_by_fold.csv", index=False)
    return summary


def plot_validation_scheme(train: pd.DataFrame, valid: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150)
    month_start = lambda m: pd.Timestamp(2025, m, 1)
    month_end = lambda m: pd.Timestamp(2025, m, 1) + pd.offsets.MonthEnd(1)
    for i, (a, b) in enumerate(FOLDS):
        ax.barh(i, (month_end(a) - month_start(1)).days, left=month_start(1), color="#064A56", height=0.55)
        ax.barh(i, (month_end(b) - month_end(a)).days, left=month_end(a), color="#E67E22", height=0.55)
    ax.barh(len(FOLDS), (train.date.max() - train.date.min()).days, left=train.date.min(), color="#064A56", height=0.55)
    ax.barh(len(FOLDS), (valid.date.max() - valid.date.min()).days, left=valid.date.min(), color="#C0392B", height=0.55)
    ax.set_yticks(range(len(FOLDS) + 1))
    ax.set_yticklabels([f"CV fold {i+1}" for i in range(len(FOLDS))] + ["Final model"])
    ax.invert_yaxis()
    ax.set_title("Forward-in-time validation: always train on the past, test on the following ~2 months",
                 loc="left", fontweight="bold", fontsize=11)
    ax.set_xlabel("teal = train      orange = CV test window      red = real validation set (Nov-Dec, labels hidden)",
                  fontsize=9, color="#455A60")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------
# December chart rows need features that the CSV does not contain
# ---------------------------------------------------------------------------
def complete_december_inputs(december: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Add lat/lon (from the city) and market_index (from the daily average in validation.csv)."""
    frames = pd.concat([train, valid])
    lookup = {}
    for kind in ("pickup", "delivery"):
        for city, la, lo in frames[[kind, f"{kind}_lat", f"{kind}_lon"]].drop_duplicates().itertuples(index=False):
            lookup[(kind, city)] = (la, lo)       # each city has exactly one coordinate pair (checked in EDA)
    out = december.copy()
    for kind in ("pickup", "delivery"):
        coords = out[kind].map(lambda c: lookup[(kind, c)])
        out[f"{kind}_lat"] = [c[0] for c in coords]
        out[f"{kind}_lon"] = [c[1] for c in coords]
    return out


def main(data_dir: str, out_dir: str, skip_cv: bool) -> None:
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)
    Path("figures").mkdir(exist_ok=True)
    train, valid, template, december = load_data(data_dir)

    # one clean market level per day, using features from train AND validation (no labels involved)
    dmi = daily_market_index(train, valid)
    assert dmi.index.min() <= pd.Timestamp("2025-12-01") and dmi.index.max() >= pd.Timestamp("2025-12-31")

    # ---------------- 1. validation ----------------
    if not skip_cv:
        print("=" * 78 + "\nFORWARD-IN-TIME CROSS-VALIDATION\n" + "=" * 78)
        plot_validation_scheme(train, valid, Path("figures/fig4_validation_scheme.png"))
        summary = cross_validate(train, dmi, out)
        pd.set_option("display.width", 200)
        print("\nMetrics on rows with trustworthy labels, averaged over the 3 folds "
              "(MAE/RMSE in dollars, MAPE in %).")
        print("RMSE_incl_corrupted shows RMSE if the ~1.4% corrupted labels are also counted (nobody can predict those).\n")
        print(summary.to_string())

    # ---------------- 2. final model ----------------
    print("\n" + "=" * 78 + "\nFINAL MODEL\n" + "=" * 78)
    model = FreightRateModel(dmi).fit(train)
    print(f"trained on {model.n_train_:,} of {len(train):,} rows ({len(train) - model.n_train_:,} corrupted labels removed)")
    lin_names = ["log_distance", "log_distance^2", "flatbed", "reefer", "daily_market_index", "trend(per 100d)", "quarter_end_ramp"]
    coefs = model.linear_.coef_ / model.sd_
    print("linear coefficients (per unit of raw feature, log-rate units):")
    for n, c in zip(lin_names, coefs):
        print(f"   {n:20s} {c:+.4f}")

    # ---------------- 3. validation predictions ----------------
    pred = model.predict(valid)
    result = pd.DataFrame({"load_id": valid.load_id.values, "predicted_rate": np.round(pred, 2)})
    result = template[["load_id"]].merge(result, on="load_id", how="left")      # keep the template's row order
    assert result.predicted_rate.notna().all() and (result.predicted_rate > 0).all() and len(result) == 12_000
    result.to_csv(out / "validation_predictions.csv", index=False)
    print(f"\nsaved {out/'validation_predictions.csv'}  ({len(result):,} rows)")
    print("predicted rate summary:", result.predicted_rate.describe().round(1).to_dict())

    # ---------------- 4. December chart ----------------
    dec_in = complete_december_inputs(december, train, valid)
    dec_pred = model.predict(dec_in)
    december_out = december.copy()
    december_out["predicted_rate"] = np.round(dec_pred, 2)
    december_out["date"] = december_out["date"].dt.strftime("%Y-%m-%d")
    december_out.to_csv(out / "december_chart_inputs.csv", index=False)
    print(f"saved {out/'december_chart_inputs.csv'}")
    print(december_out[["date", "predicted_rate"]].iloc[[0, 1, 6, 7, 14, 21, 28, 29, 30]].to_string(index=False))
    print("\nNext:  python score.py --predictions outputs/validation_predictions.csv "
          "--december-predictions outputs/december_chart_inputs.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--skip-cv", action="store_true", help="skip validation, only train + predict")
    a = ap.parse_args()
    main(a.data_dir, a.out_dir, a.skip_cv)
