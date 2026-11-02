"""
common.py - everything the other scripts share.

    load_data()            read the CSVs
    clean_inputs()         fix data-quality problems in the FEATURES
    daily_market_index()   de-noise market_index into one value per day
    flag_label_outliers()  find corrupted TARGET values (posted_rate)
    FreightRateModel       the final model (linear structure + gradient boosting)

THE ONE-PARAGRAPH SUMMARY OF THE MODEL
--------------------------------------
We predict log(posted_rate).  In log space almost everything is additive:

    log(rate) =  f(distance) + equipment + weight + geography      (what the GBM learns)
               + 0.14 * daily_market_index                          (market level that day)
               + 0.0002 * days_since_Jan_1                          (slow upward drift)
               + 0.035 * quarter_end_ramp                           (rates climb in the last month of each quarter)
               + noise

The "time" terms are put in a plain linear model on purpose: a tree can never
predict outside the range it saw in training, but the validation set is in the
FUTURE (Nov-Dec) so the time terms must extrapolate.  Everything that is not
about time is handled by a gradient-boosted tree model fitted on the leftovers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
T0 = pd.Timestamp("2025-01-01")        # day 0 of the "days since" trend feature
RAMP_DAYS = 31                         # length of the quarter-end ramp (tuned in CV: 28-35 all fine)
OUTLIER_LOG_THRESHOLD = 0.40           # |log-residual| above this => corrupted label (clear gap 0.4-0.7 in data)
EQUIPMENT_CODE = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
COORD_COLS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


# ----------------------------------------------------------------------------
# 1. loading
# ----------------------------------------------------------------------------
def load_data(data_dir: str | Path = "data"):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(data_dir / "validation.csv", parse_dates=["date"])
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    december = pd.read_csv(data_dir / "december_chart_inputs.csv", parse_dates=["date"])
    return train, valid, template, december


# ----------------------------------------------------------------------------
# 2. feature-side data quality
# ----------------------------------------------------------------------------
def clean_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Fix problems in the input columns (never touches the target).

    * weight < 0   : a sign error (292 train / 145 validation rows).  |weight| is
                     exactly the same distribution as the positive rows, so we
                     take the absolute value.
    * weight NaN   : ~0.6% missing at random, left as NaN (the GBM handles NaN
                     natively and the residuals of those rows look normal).
    * weight 47500 : the column is capped at 47,500 (a real-world legal limit
                     and a clear spike in the histogram); kept as is.
    * market_index NaN : ~0.8% missing; we never use the raw per-row value, only a
                     daily average, so the missing rows simply don't vote.
    """
    out = df.copy()
    out["weight"] = out["weight"].abs()
    return out


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """One market_index value per calendar day.

    The row-level market_index is the *same* daily level plus noise (sd ~0.025 per
    row, ~150-200 rows per day).  Averaging over a day gives the clean level.
    Only FEATURES are used here (no labels), so it is legal to average over
    train + validation together - that is how we get a value for Nov/Dec.
    """
    all_rows = pd.concat([f[["date", "market_index"]] for f in frames])
    return all_rows.groupby("date")["market_index"].mean().rename("daily_market_index")


# ----------------------------------------------------------------------------
# 3. features
# ----------------------------------------------------------------------------
def days_to_quarter_end(dates: pd.Series) -> np.ndarray:
    quarter_end = dates.dt.to_period("Q").dt.end_time.dt.normalize()
    return (quarter_end - dates).dt.days.to_numpy()


def linear_features(df: pd.DataFrame, dmi: pd.Series, use_trend=True, use_ramp=True, use_market=True) -> np.ndarray:
    """Columns for the linear part: the things that must EXTRAPOLATE in time."""
    log_d = np.log(df["distance"].to_numpy())
    cols = [
        log_d,
        log_d ** 2,                                   # rate-per-mile falls with distance -> curvature
        (df["equipment"] == "Flatbed").to_numpy(float),
        (df["equipment"] == "Reefer").to_numpy(float),
    ]
    if use_market:
        cols.append(df["date"].map(dmi).to_numpy(float))
    if use_trend:
        cols.append((df["date"] - T0).dt.days.to_numpy(float) / 100.0)
    if use_ramp:
        cols.append(np.clip(1.0 - days_to_quarter_end(df["date"]) / RAMP_DAYS, 0.0, 1.0))
    return np.column_stack(cols).astype(float)


def tree_features(df: pd.DataFrame) -> pd.DataFrame:
    """Columns for the GBM: shape of the price surface that is NOT about time."""
    return pd.DataFrame(
        {
            "log_distance": np.log(df["distance"]),
            "equipment": df["equipment"].map(EQUIPMENT_CODE),
            "weight": df["weight"],                       # NaN allowed
            **{c: df[c] for c in COORD_COLS},             # geography -> also works for unseen cities
        },
        index=df.index,
    )


# ----------------------------------------------------------------------------
# 4. target-side data quality: corrupted labels
# ----------------------------------------------------------------------------
def flag_label_outliers(df: pd.DataFrame, dmi: pd.Series) -> np.ndarray:
    """Return a boolean array, True = label looks CORRECT, False = corrupted.

    ~1.4% of posted_rate values are off by a factor of roughly 2-6x (up or down).  On the
    log scale a simple model's residuals are tiny (sd ~0.04) for the 98.6% good
    rows and |residual| > 0.7 for the bad ones - nothing in between - so a
    threshold of 0.4 separates them cleanly.  We fit twice so the outliers do not
    bend the first fit.
    """
    y = np.log(df["posted_rate"].to_numpy())
    X = np.column_stack([np.ones(len(df)), linear_features(df, dmi)])
    keep = np.ones(len(df), dtype=bool)
    for threshold in (0.5, OUTLIER_LOG_THRESHOLD):
        beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        keep = np.abs(y - X @ beta) < threshold
    return keep


# ----------------------------------------------------------------------------
# 5. the model
# ----------------------------------------------------------------------------
class FreightRateModel:
    """Linear (time + market + distance) + gradient boosting on the leftovers."""

    def __init__(self, dmi: pd.Series, use_trend=True, use_ramp=True, use_market=True,
                 use_gbm=True, gbm_params: dict | None = None):
        self.dmi = dmi
        self.use_gbm = use_gbm
        self.opts = dict(use_trend=use_trend, use_ramp=use_ramp, use_market=use_market)
        self.gbm_params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                               min_samples_leaf=60, l2_regularization=1.0, random_state=0)
        if gbm_params:
            self.gbm_params.update(gbm_params)

    def fit(self, df: pd.DataFrame, drop_outliers: bool = True):
        df = clean_inputs(df)
        if drop_outliers:
            df = df[flag_label_outliers(df, self.dmi)]
        self.n_train_ = len(df)
        y = np.log(df["posted_rate"].to_numpy())

        L = linear_features(df, self.dmi, **self.opts)
        self.mu_, self.sd_ = L.mean(0), L.std(0) + 1e-9
        self.linear_ = Ridge(alpha=1e-6).fit((L - self.mu_) / self.sd_, y)   # ~ ordinary least squares
        resid = y - self.linear_.predict((L - self.mu_) / self.sd_)

        if self.use_gbm:
            self.gbm_ = HistGradientBoostingRegressor(**self.gbm_params).fit(tree_features(df), resid)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df = clean_inputs(df)
        L = (linear_features(df, self.dmi, **self.opts) - self.mu_) / self.sd_
        log_pred = self.linear_.predict(L)
        if self.use_gbm:
            log_pred = log_pred + self.gbm_.predict(tree_features(df))
        return np.exp(log_pred)        # back to dollars (the log-normal bias term is ~0.02%, ignored)
