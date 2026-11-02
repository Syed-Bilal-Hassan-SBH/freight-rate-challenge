"""
eda.py - look at the data BEFORE modelling.

Run:  python eda.py
Prints the audit to the console and saves figures to figures/.
Every printed section answers one question; the answers are what drove the modelling choices.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FreightRateModel, clean_inputs, daily_market_index, days_to_quarter_end,
                    flag_label_outliers, linear_features, load_data)

TEAL, GREY, RED, ORANGE = "#064A56", "#7F8C8D", "#C0392B", "#E67E22"


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main(data_dir: str, fig_dir: str) -> None:
    pd.set_option("display.width", 200, "display.max_columns", 30)
    Path(fig_dir).mkdir(exist_ok=True, parents=True)
    train, valid, _, december = load_data(data_dir)

    # ------------------------------------------------------------------ Q1
    section("Q1. What is in the files?  (shape, types, time span)")
    print(f"train_test : {train.shape}  dates {train.date.min().date()} -> {train.date.max().date()}")
    print(f"validation : {valid.shape}  dates {valid.date.min().date()} -> {valid.date.max().date()}")
    print("target     : posted_rate (only in train_test). Everything else is a feature.")
    print("=> The validation set is entirely AFTER the training period (Nov-Dec vs Jan-Oct).")
    print("   This is a FORECASTING problem, so validation must be forward in time, not random.")

    # ------------------------------------------------------------------ Q2
    section("Q2. Missing values / impossible values")
    print(pd.DataFrame({"train_missing": train.isna().sum(), "valid_missing": valid.isna().sum()})
          .query("train_missing > 0 or valid_missing > 0"))
    print(f"\nNegative weights   : train {(train.weight < 0).sum()}  valid {(valid.weight < 0).sum()}")
    print(f"|weight| == 47,500 : train {(train.weight.abs() == 47500).sum()}  (column is capped there)")
    print(f"weight == 0        : {(train.weight == 0).sum()}")
    print(f"duplicate load_id  : {train.load_id.duplicated().sum()}")
    print(f"pickup == delivery : {(train.pickup == train.delivery).sum()}")
    print("=> weight has sign errors and gaps; market_index has gaps.  Handled in clean_inputs().")

    # ------------------------------------------------------------------ Q3
    section("Q3. Do validation categories exist in training?  (generalisation risk)")
    seen = set(train.pickup) | set(train.delivery)
    unseen = sorted((set(valid.pickup) | set(valid.delivery)) - seen)
    share = (~(valid.pickup.isin(seen) & valid.delivery.isin(seen))).mean()
    print(f"cities in train: {len(seen)}   cities in validation: {len(set(valid.pickup) | set(valid.delivery))}")
    print(f"cities NEVER seen in training: {unseen}")
    print(f"=> {share:.1%} of validation rows touch an unseen city.  A model that memorises city names (one-hot /"
          "\n   target encoding) would fail on them -> use lat/lon + distance instead.  I simulate this in CV.")
    print("\nEquipment mix (train):", train.equipment.value_counts(normalize=True).round(3).to_dict())
    print("Equipment mix (valid):", valid.equipment.value_counts(normalize=True).round(3).to_dict())

    # ------------------------------------------------------------------ Q4
    section("Q4. How does the target behave?")
    tr = clean_inputs(train)
    tr["rpm"] = tr.posted_rate / tr.distance
    print(tr[["posted_rate", "distance", "rpm"]].describe().round(2))
    print("\nSpearman correlation with posted_rate:")
    print(tr[["posted_rate", "distance", "weight", "market_index", "quote_signal"]].corr("spearman")["posted_rate"].round(3))
    print("=> rate is driven almost entirely by distance (rho=0.98), roughly log-log.  Rate-per-mile is NOT constant:"
          "\n   short hauls cost more per mile, so we use log(distance) and log(distance)^2.")
    print("\nRate per mile by equipment (median):", tr.groupby("equipment").rpm.median().round(3).to_dict())

    # ------------------------------------------------------------------ Q5
    section("Q5. Are there corrupted labels?")
    dmi = daily_market_index(train, valid)
    keep = flag_label_outliers(tr, dmi)
    y = np.log(tr.posted_rate.to_numpy())
    X = np.column_stack([np.ones(len(tr)), linear_features(tr, dmi)])
    beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    resid = y - X @ beta
    print(f"rows flagged as corrupted : {(~keep).sum()}  ({(~keep).mean():.2%})")
    print(f"  too high (rate x2-6)    : {(resid[~keep] > 0).sum()}")
    print(f"  too low  (rate /2-6)    : {(resid[~keep] < 0).sum()}")
    print(f"residual sd, good rows    : {resid[keep].std():.4f} (log units, ~{resid[keep].std()*100:.1f}%)")
    print(f"smallest |resid| among flagged: {np.abs(resid[~keep]).min():.2f}  (clear gap -> threshold 0.4 is safe)")
    print(tr.loc[~keep, ["pickup", "delivery", "distance", "equipment", "posted_rate", "rpm"]]
          .sort_values("rpm").iloc[[0, 1, 2, -3, -2, -1]])

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.hist(resid, bins=120, color=TEAL)
    ax.set_yscale("log")
    ax.axvline(0.4, color=RED, ls="--"); ax.axvline(-0.4, color=RED, ls="--")
    ax.set_title("Label check: residual of a simple log-model (log scale on y)", loc="left", fontweight="bold")
    ax.set_xlabel("log(actual rate) - log(model rate)"); ax.set_ylabel("rows")
    ax.text(0.42, ax.get_ylim()[1] * 0.35, "corrupted labels\n(~1.4%)", color=RED, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig1_label_outliers.png"); plt.close(fig)

    # ------------------------------------------------------------------ Q6
    section("Q6. How do the two 'market' columns behave over time?  (incl. the validation period)")
    daily = pd.concat([train, valid]).groupby("date").agg(dmi=("market_index", "mean"), dqs=("quote_signal", "mean"))
    print("market_index within a day: sd ~0.025 per row vs. a yearly range 0.7-1.4  => it is ONE daily level + noise.")
    print(f"market_index autocorrelation: lag1={daily.dmi.autocorr(1):.2f}  lag7={daily.dmi.autocorr(7):.2f}  (weekly cycle)")
    m = daily.groupby(daily.index.month).agg(["mean", "std"]).round(3)
    print("\nMonthly mean (std of the daily values):")
    print(m)
    print("=> quote_signal changes REGIME month to month (2.3 in June, 1.9 in July, 2.05 in Nov/Dec with tiny spread).")
    print("   It looks like a date fingerprint, not a stable driver.  See modelling notes for why it is excluded.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=150, sharex=True)
    for ax, col, color, label in [(axes[0], "dmi", TEAL, "daily mean market_index"),
                                  (axes[1], "dqs", ORANGE, "daily mean quote_signal")]:
        ax.plot(daily.index, daily[col], color=color, lw=1.6)
        ax.axvspan(valid.date.min(), valid.date.max(), color=GREY, alpha=0.15)
        ax.set_ylabel(label, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(valid.date.min(), axes[0].get_ylim()[1] * 0.97, " validation period", va="top", color=GREY, fontsize=9)
    axes[0].set_title("Daily market signals, Jan-Dec 2025", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig2_market_signals.png"); plt.close(fig)

    # ------------------------------------------------------------------ Q7
    section("Q7. What explains the day-to-day level of rates?  (the key discovery)")
    clean_rows = tr[keep].copy()
    # model WITHOUT the quarter-end ramp, so the ramp shows up in the leftovers
    m0 = FreightRateModel(dmi, use_ramp=False).fit(train, drop_outliers=True)
    clean_rows["resid"] = np.log(clean_rows.posted_rate) - np.log(m0.predict(clean_rows))
    clean_rows["dtq"] = days_to_quarter_end(clean_rows.date)
    by_dtq = clean_rows.groupby("dtq").resid.mean()
    print("Mean leftover (log units) by days until quarter-end, Q1-Q3 data:")
    print(by_dtq.groupby(pd.cut(by_dtq.index, [-1, 0, 2, 5, 9, 14, 21, 30, 45, 60, 100])).mean().round(4))
    print("=> A steady climb of about +3.5% over the last month of every quarter, then a reset on day 1 of the next."
          "\n   This is a CALENDAR effect, so it is modelled with a days_to_quarter_end ramp feature."
          "\n   Consequence for the December chart: rates should rise through December towards Dec 31.")

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.scatter(by_dtq.index, by_dtq.values, s=14, color=TEAL)
    ax.axhline(0, color=GREY, lw=0.8); ax.invert_xaxis()
    ax.set_title("Quarter-end effect: leftover rate level vs. days until quarter end", loc="left", fontweight="bold")
    ax.set_xlabel("days until quarter end  (0 = last day of Mar / Jun / Sep)"); ax.set_ylabel("mean log-leftover")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig3_quarter_end_ramp.png"); plt.close(fig)

    # ------------------------------------------------------------------ Q8
    section("Q8. What do the December chart inputs need?")
    print(december.head(3))
    print("\nIt only has pickup, delivery, distance, equipment, weight, date.  The model also needs lat/lon (looked up"
          "\nfrom the city) and the daily market index (averaged from validation.csv rows on that date) - see train_and_predict.py.")
    print("\nEDA finished.  Figures saved in:", fig_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--fig-dir", default="figures")
    a = ap.parse_args()
    main(a.data_dir, a.fig_dir)
