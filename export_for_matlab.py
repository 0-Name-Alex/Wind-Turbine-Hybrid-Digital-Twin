"""
export_for_matlab.py
--------------------
Exports Wind Farm A (CARE to Compare) into clean CSVs for the MATLAB/Simulink
physics model (Part 1 + Part 2 of the RET digital twin).

What it does
  * Reads event_info.csv and every dataset listed in it (22 files).
  * Keeps only the signals the physics model needs (see SIGNALS below)
    and renames them to readable names.
  * Adds two columns that are for EVALUATION ONLY (never model inputs):
      - status_type_id  (logbook-based -> leakage if used as input)
      - in_event        (1 inside the labelled fault window, else 0)
  * Writes one CSV per dataset to  matlab_data/  plus  export_summary.csv
    (min / median / max of every signal, so we can check units and scaling).

Usage (Anaconda Prompt, env "ret", inside the project folder):
    python export_for_matlab.py
    python export_for_matlab.py --data "path\\to\\Wind Farm A"   (if auto-find fails)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# CARE sensor -> readable name used in MATLAB
SIGNALS = {
    "wind_speed_3_avg": "wind",            # m/s
    "sensor_0_avg":     "T_amb",           # ambient temperature
    "sensor_43_avg":    "T_nacelle",       # nacelle temperature
    "sensor_5_avg":     "pitch",           # deg
    "sensor_52_avg":    "rotor_rpm",
    "sensor_18_avg":    "gen_rpm",
    "power_30_avg":     "P",               # grid power
    "sensor_12_avg":    "T_gb_oil",        # gearbox oil
    "sensor_11_avg":    "T_gb_hss",        # gearbox bearing, high-speed shaft
    "sensor_13_avg":    "T_genbrg_DE",     # generator bearing 2 (drive end)
    "sensor_14_avg":    "T_genbrg_NDE",    # generator bearing 1 (non-drive end)
    "sensor_15_avg":    "T_stator1",
    "sensor_16_avg":    "T_stator2",
    "sensor_17_avg":    "T_stator3",
    "sensor_38_avg":    "T_trafo1",
    "sensor_39_avg":    "T_trafo2",
    "sensor_40_avg":    "T_trafo3",
    "sensor_41_avg":    "T_hyd_oil",       # hydraulic group oil
}

EVAL_ONLY = ["status_type_id"]


def find_farm_folder(start: Path) -> Path:
    """Look for a folder containing event_info.csv + datasets/ (Wind Farm A)."""
    hits = [p.parent for p in start.rglob("event_info.csv")
            if (p.parent / "datasets").is_dir()]
    farm_a = [h for h in hits if "a" in h.name.lower().split()[-1:]]
    if farm_a:
        return farm_a[0]
    if hits:
        return hits[0]
    raise FileNotFoundError(
        "Could not find Wind Farm A automatically. Run with --data \"<folder "
        "that contains event_info.csv and datasets>\"")


def find_col(df: pd.DataFrame, name: str):
    """Exact match first, then case-insensitive match."""
    if name in df.columns:
        return name
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="Folder with event_info.csv and datasets/")
    ap.add_argument("--out", type=str, default="matlab_data")
    args = ap.parse_args()

    farm = Path(args.data) if args.data else find_farm_folder(Path.cwd())
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    print(f"Wind Farm A folder : {farm}")
    print(f"Output folder      : {out.resolve()}\n")

    events = pd.read_csv(farm / "event_info.csv", sep=";")
    events["event_start"] = pd.to_datetime(events["event_start"])
    events["event_end"] = pd.to_datetime(events["event_end"])

    all_frames, missing_report = [], {}

    for _, ev in events.sort_values(["asset", "event_id"]).iterrows():
        f = farm / "datasets" / f"{int(ev.event_id)}.csv"
        if not f.exists():
            print(f"  !! missing dataset file {f.name}, skipped")
            continue
        df = pd.read_csv(f, sep=";")

        ts_col = find_col(df, "time_stamp")
        t = pd.to_datetime(df[ts_col])
        res = pd.DataFrame({"time": t.dt.strftime("%Y-%m-%d %H:%M:%S")})

        tt = find_col(df, "train_test")
        res["train_test"] = df[tt].values if tt else "unknown"

        missing = []
        for src, dst in SIGNALS.items():
            c = find_col(df, src)
            if c is None:
                missing.append(src)
                res[dst] = np.nan
            else:
                res[dst] = pd.to_numeric(df[c], errors="coerce").values
        if missing:
            missing_report[int(ev.event_id)] = missing

        # evaluation-only columns (NOT model inputs)
        for e in EVAL_ONLY:
            c = find_col(df, e)
            res[e] = df[c].values if c else np.nan
        res["in_event"] = ((t >= ev.event_start) & (t <= ev.event_end)).astype(int).values

        label = "fault" if ev.event_label == "anomaly" else "healthy"
        desc = str(ev.event_description) if isinstance(ev.event_description, str) else ""
        name = f"A{int(ev.asset):02d}_ev{int(ev.event_id):02d}_{label}.csv"
        res.to_csv(out / name, index=False)
        print(f"  {name:28s} rows={len(res):6d}  {desc}")

        tmp = res.copy()
        tmp["asset"], tmp["event_id"], tmp["label"] = int(ev.asset), int(ev.event_id), label
        all_frames.append(tmp)

    # ---- summary: units / scaling check -------------------------------------
    big = pd.concat(all_frames, ignore_index=True)
    rows = []
    for col in SIGNALS.values():
        s = big[col].dropna()
        rows.append({"signal": col,
                     "min": s.min() if len(s) else np.nan,
                     "p50": s.median() if len(s) else np.nan,
                     "p99": s.quantile(0.99) if len(s) else np.nan,
                     "max": s.max() if len(s) else np.nan,
                     "missing_%": 100 * big[col].isna().mean()})
    summ = pd.DataFrame(rows)

    # gear ratio estimate (generator rpm / rotor rpm while running)
    run = big[(big.rotor_rpm > 3) & (big.gen_rpm > 100)]
    ratio = (run.gen_rpm / run.rotor_rpm).median() if len(run) else np.nan

    summ.to_csv(out / "export_summary.csv", index=False)
    pd.Series({"gear_ratio_estimate": ratio}).to_csv(out / "gear_ratio.csv")

    print("\nSignal ranges (check units / scaling):")
    print(summ.round(3).to_string(index=False))
    print(f"\nEstimated gear ratio (gen rpm / rotor rpm): {ratio:.1f}")
    if missing_report:
        print("\nMissing columns per dataset:", missing_report)
    print("\nDone. Upload export_summary.csv, gear_ratio.csv and 2-3 of the "
          "CSVs (e.g. one healthy, one gearbox fault) to continue.")


if __name__ == "__main__":
    main()
