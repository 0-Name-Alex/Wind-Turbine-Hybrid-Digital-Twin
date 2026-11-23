"""
Health trend & time-to-alarm (Digital Twin - future prediction)
===============================================================
For every subsystem, the twin fits a straight line to its smoothed anomaly
score over the last 24 h and projects when it would cross the alarm threshold.

A "trend warning" is raised when, for at least 1 hour,
    - a subsystem score is above 50 % of the threshold, and
    - its projected crossing is within the next 24 h.

This script checks on all 22 datasets whether trend warnings come EARLIER than
the alarms (anomaly datasets) and how often they appear on healthy turbines.

It is an early-warning indicator, NOT a remaining-useful-life model
(12 failures are far too few to learn time-to-failure).

Needs: outputs_fault_detection/scores/*.csv  (run fault_detection.py first)
Output: outputs_fault_detection/health_trend_results.csv, health_trend_summary.json
Usage:  python health_trend.py [--horizon-hours 24] [--window-hours 24]
"""
import os
# hide harmless sklearn/joblib version warnings, also in the parallel worker processes
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse, glob, json, os
import numpy as np
import pandas as pd
from fault_detection import health_trend, SUBSYSTEMS, STEPS_PER_HOUR


def first_true_run(mask, min_len):
    m = np.asarray(mask, bool); run = 0
    for i, v in enumerate(m):
        run = run + 1 if v else 0
        if run >= min_len:
            return i - min_len + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores")
    ap.add_argument("--horizon-hours", type=float, default=24)
    ap.add_argument("--window-hours", type=float, default=24)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sdir = a.scores or os.path.join(here, "outputs_fault_detection", "scores")
    rows = []
    for f in sorted(glob.glob(os.path.join(sdir, "*.csv")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0])):
        ds = int(os.path.splitext(os.path.basename(f))[0])
        d = pd.read_csv(f, parse_dates=["time_stamp"])
        subs = [c for c in SUBSYSTEMS if c in d]
        thr = float(d["threshold"].iloc[0])
        _, hrs = health_trend(d[subs], thr, int(a.window_hours * STEPS_PER_HOUR))
        warn = ((hrs <= a.horizon_hours) & (d[subs] > 0.5 * thr)).any(axis=1)
        win = d["in_event_window"].astype(bool)
        is_anom = bool(win.any())
        r = {"dataset_id": ds, "label": "anomaly" if is_anom else "normal"}
        if is_anom:
            ev_start = d.loc[win, "id"].min()
            i_w = first_true_run(warn & win, STEPS_PER_HOUR)
            i_a = first_true_run(d["alarm"].astype(bool) & win, 1)
            r["trend_warning_in_window"] = i_w is not None
            r["alarm_in_window"] = i_a is not None
            r["early_trend_warning_before_window"] = first_true_run(warn & (d["id"] < ev_start), STEPS_PER_HOUR) is not None
            if i_w is not None and i_a is not None:
                r["hours_earlier_than_alarm"] = round((d["time_stamp"][i_a] - d["time_stamp"][i_w]).total_seconds() / 3600, 1)
            if i_w is not None:
                r["trend_subsystem"] = hrs.loc[i_w, subs].idxmin()
        else:
            r["false_trend_warning"] = first_true_run(warn, STEPS_PER_HOUR) is not None
        rows.append(r)
    res = pd.DataFrame(rows)
    out = os.path.dirname(sdir)
    res.to_csv(os.path.join(out, "health_trend_results.csv"), index=False)
    an, no = res[res.label == "anomaly"], res[res.label == "normal"]
    both = an.dropna(subset=["hours_earlier_than_alarm"]) if "hours_earlier_than_alarm" in an else an.iloc[0:0]
    summ = {"anomaly_datasets": len(an), "normal_datasets": len(no),
            "faults_with_trend_warning": int(an.get("trend_warning_in_window", pd.Series(dtype=bool)).sum()),
            "faults_with_alarm": int(an.get("alarm_in_window", pd.Series(dtype=bool)).sum()),
            "median_hours_trend_before_alarm": float(both["hours_earlier_than_alarm"].median()) if len(both) else None,
            "normal_datasets_with_trend_warning": int(no.get("false_trend_warning", pd.Series(dtype=bool)).sum()),
            "settings": vars(a)}
    json.dump(summ, open(os.path.join(out, "health_trend_summary.json"), "w"), indent=2, default=str)
    print(res.to_string(index=False))
    print("\n=========== HEALTH TREND SUMMARY ===========")
    print(f"Faults with a trend warning      : {summ['faults_with_trend_warning']}/{len(an)}  (alarms: {summ['faults_with_alarm']}/{len(an)})")
    print(f"Median hours earlier than alarm  : {summ['median_hours_trend_before_alarm']}")
    print(f"Healthy turbines with a warning  : {summ['normal_datasets_with_trend_warning']}/{len(no)}")


if __name__ == "__main__":
    main()
