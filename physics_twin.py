"""
physics_twin.py - physics model of the turbine for the hybrid digital twin
==========================================================================
Same equations as the MATLAB/Simulink physics twin (matlab/ folder):

  Rotor + control  P = Kp * Cp(lambda, 0) * v^3, capped at rated power   (wt_aero.m / wt_power_vec.m)
                   rotor speed follows the best tip-speed ratio (MPPT)
  Losses -> heat   each part's "equilibrium" temperature is a sum of loss terms
                   C * [P, P^2, w, T_nacelle, T_ambient, 1]                (wt_heat.m)
  Heat -> temp.    first-order lag  T[k] = T[k-1] + a (T_eq[k] - T[k-1]),  a = 600 s / tau

How it is used in the dashboard (measured-load mode, as run_twin(..., 'Load','measured')):
  expected temperature = physics model driven by the MEASURED power, rotor speed and air
  temperatures. gap = measured - expected. The daily mean of the gap is compared with
  the normal daily gap on healthy data (the "band").

Tuning (same method as calibrate_twin.m): for each turbine, on its most recent
CAL_DAYS of healthy data. Test periods and logged fault windows of that turbine are
left out, so the twin never tunes on the periods it is judged on.
Results are cached in outputs_physics/params_A<asset>.json.

Physics health of a part (0-100 %, same colours as the learning model):
  green while the daily gap stays inside the normal band, amber between 1 and 2 bands,
  red above 2 bands (see gap_to_health)
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter

TS = 600.0                      # 10-min step (s)
RATED_RPM = 14.9                # rotor speed at rated (Wind Farm A)
CAL_DAYS = 240                  # tune on the most recent 8 months of healthy data
DAY = 144                       # 10-min steps per day
PARAMS_VERSION = 1
LOW_LOAD = 0.03                 # daily mean power below 3 % of rated = turbine (nearly) stopped

SIGNALS = {"wind_speed_3_avg": "wind", "sensor_0_avg": "T_amb", "sensor_43_avg": "T_nacelle",
           "power_30_avg": "P", "sensor_52_avg": "rotor_rpm",
           "sensor_12_avg": "T_gb_oil", "sensor_11_avg": "T_gb_hss",
           "sensor_13_avg": "T_genbrg_DE", "sensor_14_avg": "T_genbrg_NDE",
           "sensor_15_avg": "T_stator1", "sensor_40_avg": "T_trafo3"}

TERMS = ["P", "P2", "w", "T_nac", "T_amb", "one"]
TARGETS = {   # temperature: (readable name, loss terms used)   - same choice as calibrate_twin.m
    "T_gb_oil":     ("gearbox oil",                   [1, 0, 1, 1, 1, 1]),
    "T_gb_hss":     ("gearbox high-speed bearing",    [1, 0, 1, 1, 1, 1]),
    "T_genbrg_DE":  ("generator bearing (drive end)", [0, 1, 1, 1, 1, 1]),
    "T_genbrg_NDE": ("generator bearing (non-drive end)", [0, 1, 1, 1, 1, 1]),
    "T_stator1":    ("generator winding",             [0, 1, 1, 1, 1, 1]),
    "T_trafo3":     ("transformer",                   [0, 1, 0, 1, 1, 1]),
}
PHYS_PARTS = {"gearbox": ["T_gb_oil", "T_gb_hss"],
              "generator": ["T_genbrg_DE", "T_genbrg_NDE", "T_stator1"],
              "transformer": ["T_trafo3"]}


# ------------------------------------------------------------------ data
def read_raw(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        sep = ";" if f.readline().count(";") > 1 else ","
    keep = {"time_stamp", "train_test", "status_type_id", *SIGNALS}
    d = pd.read_csv(path, sep=sep, usecols=lambda c: c in keep)
    d["time_stamp"] = pd.to_datetime(d["time_stamp"])
    d = d.rename(columns=SIGNALS).drop_duplicates("time_stamp").sort_values("time_stamp").reset_index(drop=True)
    for c in SIGNALS.values():
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in TARGETS:
        d.loc[d[c] > 200, c] = np.nan                     # 205 degC = sensor error code
    for c in ["wind", "T_amb", "T_nacelle", "P", "rotor_rpm"]:
        d[c] = d[c].interpolate(limit=6).bfill().ffill()
    d["w"] = d["rotor_rpm"] / RATED_RPM
    return d


def _segments(t):
    return np.cumsum(np.r_[True, np.diff(t.values) != np.timedelta64(10, "m")])


def _regressors(d):
    p = d["P"].clip(lower=0).to_numpy()
    return np.c_[p, p ** 2, d["w"].to_numpy(), d["T_nacelle"].to_numpy(), d["T_amb"].to_numpy(), np.ones(len(d))]


def _lag(X, a, seg):
    """first-order thermal lag, restarted at data gaps"""
    Y = np.zeros_like(X, dtype=float)
    edges = np.r_[np.flatnonzero(np.r_[True, np.diff(seg) != 0]), len(seg)]
    for s in range(len(edges) - 1):
        i = slice(edges[s], edges[s + 1])
        Y[i] = lfilter([a], [1, -(1 - a)], X[i], axis=0)
    return Y


def _start(y, a, seg):
    """decay of the measured start temperature of each block (the model's initial state)"""
    r = np.zeros(len(y))
    edges = np.r_[np.flatnonzero(np.r_[True, np.diff(seg) != 0]), len(seg)]
    fallback = np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 40.0
    for s in range(len(edges) - 1):
        i = np.arange(edges[s], edges[s + 1])
        blk = y[i]
        y0 = blk[0] if np.isfinite(blk[0]) else (np.nanmean(blk) if np.isfinite(blk).any() else fallback)
        r[i] = y0 * (1 - a) ** np.arange(1, len(i) + 1)
    return r


# ------------------------------------------------------------------ power (Cp - lambda)
def cp(lam, beta=0.0):
    li = 1.0 / (1.0 / (lam + 0.08 * beta) - 0.035 / (beta ** 3 + 1))
    return np.maximum(0.5176 * (116 / li - 0.4 * beta - 5) * np.exp(-21 / li) + 0.0068 * lam, 0)


def power(v, Kp, R, wmin, Prated, Pstandby=-0.004):
    v = np.asarray(v, float)
    w_rated = RATED_RPM * 2 * np.pi / 60
    w = np.clip(8.1 * v / R, wmin * w_rated, w_rated)
    lam = w * R / np.maximum(v, 0.1)
    P = np.minimum(Kp * cp(lam) * v ** 3, Prated)
    return np.where((v < 3.5) | (v > 25), Pstandby, P)


# ------------------------------------------------------------------ tuning
def _healthy_files(farm_dir, events, asset):
    ids = events[(events["asset"] == asset) & (events["event_label"] == "normal")]["event_id"]
    return [os.path.join(farm_dir, "datasets", f"{int(i)}.csv") for i in ids
            if os.path.isfile(os.path.join(farm_dir, "datasets", f"{int(i)}.csv"))]


def calibrate(asset, farm_dir):
    events = pd.read_csv(os.path.join(farm_dir, "event_info.csv"), sep=";")
    events["event_start"] = pd.to_datetime(events["event_start"])
    events["event_end"] = pd.to_datetime(events["event_end"])
    files = _healthy_files(farm_dir, events, asset)
    if not files:
        raise FileNotFoundError(f"no healthy datasets for turbine {asset}")
    D = pd.concat([read_raw(f) for f in files]).drop_duplicates("time_stamp").sort_values("time_stamp")
    D = D[D["train_test"] == "train"]
    # leave out every test period and logged fault window of this turbine (+/- 1 day)
    keep = np.ones(len(D), bool)
    for _, e in events[events["asset"] == asset].iterrows():
        keep &= ~((D["time_stamp"] >= e["event_start"] - pd.Timedelta(days=1)) &
                  (D["time_stamp"] <= e["event_end"] + pd.Timedelta(days=1))).to_numpy()
    D = D[keep]
    D = D[D["time_stamp"] >= D["time_stamp"].max() - pd.Timedelta(days=CAL_DAYS)].reset_index(drop=True)
    seg = _segments(D["time_stamp"])

    p = {"version": PARAMS_VERSION, "asset": int(asset), "cal_from": str(D["time_stamp"].min()),
         "cal_to": str(D["time_stamp"].max()), "cal_rows": int(len(D)), "Pstandby": -0.004}
    # power curve (normal operation only, robust loss ignores curtailed points)
    n = (D["status_type_id"] == 0) & D["wind"].notna() & D["P"].notna()
    hi = n & (D["wind"] > 16)
    p["Prated"] = float(D.loc[hi, "P"].median()) if hi.sum() > 50 else 0.975
    v, y = D.loc[n, "wind"].to_numpy(), D.loc[n, "P"].to_numpy()
    fit = least_squares(lambda q: power(v, q[0], q[1], q[2], p["Prated"]) - y, x0=[1.6e-3, 43, 0.67],
                        bounds=([1e-5, 20, 0.2], [1e-1, 80, 0.9]), loss="soft_l1", f_scale=0.03)
    p["Kp"], p["R"], p["wmin"] = map(float, fit.x)
    e = y - power(v, p["Kp"], p["R"], p["wmin"], p["Prated"])
    p["P_r2"] = float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))

    X = _regressors(D)
    p["targets"] = {}
    for t, (_, use) in TARGETS.items():
        y = D[t].to_numpy(float)
        if np.isfinite(y).sum() < 2000:
            continue
        u = np.array(use, bool)
        best = None
        for th in [0.5, 1, 2, 3, 4, 6, 8, 12]:
            a = TS / (th * 3600)
            B = _lag(X[:, u], a, seg)
            base = _start(y, a, seg)
            m = np.isfinite(y)
            c, *_ = np.linalg.lstsq(B[m], (y - base)[m], rcond=None)
            res = y - (base + B @ c)
            r2 = 1 - np.nansum(res ** 2) / np.nansum((y - np.nanmean(y)) ** 2)
            if best is None or r2 > best[0]:
                best = (r2, th, c, res)
        r2, th, c, res = best
        coef = np.zeros(6)
        coef[u] = c
        day = pd.Series(res).rolling(DAY, min_periods=DAY // 4).mean()
        p["targets"][t] = {"tau_h": th, "C": coef.tolist(), "r2": float(r2),
                           "rmse": float(np.sqrt(np.nanmean(res ** 2))),
                           "band": float(max(np.nanpercentile(day, 99), 1.0))}
    return p


def get_params(asset, farm_dir, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"params_A{int(asset):02d}.json")
    if os.path.isfile(path):
        with open(path) as f:
            p = json.load(f)
        if p.get("version") == PARAMS_VERSION:
            return p
    p = calibrate(asset, farm_dir)
    with open(path, "w") as f:
        json.dump(p, f, indent=1)
    return p


# ------------------------------------------------------------------ run the twin on one dataset
def gap_to_health(x):
    """x = daily gap / normal band.  <= 0.5 band -> 100 %, 1 band -> 70 % (edge of normal),
    2 bands -> 30 % (red), 3 bands -> 0 %.  Running colder than expected counts as healthy."""
    x = np.asarray(x, float)
    h = np.where(x <= 0.5, 100.0,
        np.where(x <= 1, 100 - 60 * (x - 0.5),
        np.where(x <= 2, 70 - 40 * (x - 1), 30 - 30 * (x - 2))))
    return np.where(np.isfinite(x), np.clip(h, 0, 100), np.nan)



def run(raw_path, p):
    """Expected temperatures, gaps and physics health for every 10-min step of one dataset."""
    d = read_raw(raw_path)
    seg = _segments(d["time_stamp"])
    X = _regressors(d)
    out = pd.DataFrame({"time_stamp": d["time_stamp"]})
    out["P_meas"] = d["P"]
    out["P_phys"] = power(d["wind"].to_numpy(), p["Kp"], p["R"], p["wmin"], p["Prated"], p["Pstandby"])
    # almost no power over the last day -> parts hardly heat up, so a heat check says little
    low = pd.Series(d["P"].clip(lower=0)).rolling(DAY, min_periods=DAY // 4).mean().to_numpy() < LOW_LOAD
    out["low_load"] = low
    for t, q in p["targets"].items():
        a = TS / (q["tau_h"] * 3600)
        y = d[t].to_numpy(float)
        exp = _start(y, a, seg) + _lag(X, a, seg) @ np.array(q["C"])
        gap = y - exp
        day = pd.Series(gap).rolling(DAY, min_periods=DAY // 4).mean().to_numpy()
        out[f"{t}_meas"], out[f"{t}_exp"], out[f"{t}_gap_day"] = y, exp, day
        out[f"{t}_health"] = np.where(low, np.nan, gap_to_health(day / q["band"]))
    for part, ts in PHYS_PARTS.items():
        cols = [f"{t}_health" for t in ts if f"{t}_health" in out]
        out[f"ph_{part}"] = out[cols].min(axis=1, skipna=True) if cols else np.nan
    return out


def worst_sensor(row, part, p):
    """(target, gap, band) of the sensor that is furthest above its expected value"""
    best = None
    for t in PHYS_PARTS.get(part, []):
        if t not in p["targets"]:
            continue
        g = row.get(f"{t}_gap_day", np.nan)
        if not np.isfinite(g):
            continue
        ratio = g / p["targets"][t]["band"]
        if best is None or ratio > best[3]:
            best = (t, g, p["targets"][t]["band"], ratio)
    return best


# ------------------------------------------------------------------ combine the two opinions
def state(h):
    if h is None or not np.isfinite(h):
        return "n/a"
    return "ok" if h >= 70 else "watch" if h >= 30 else "bad"


def combine(h_ml, h_ph):
    """Combined health (vectorised). Both models must agree for red; one alone gives amber."""
    h_ml, h_ph = np.asarray(h_ml, float), np.asarray(h_ph, float)
    both = np.isfinite(h_ml) & np.isfinite(h_ph)
    lo = np.fmin(h_ml, h_ph)
    flags = (np.nan_to_num(h_ml, nan=100) < 70).astype(int) + (np.nan_to_num(h_ph, nan=100) < 70).astype(int)
    comb = np.where(both & (flags == 1), np.maximum(lo, 35.0), lo)      # only one model worried -> amber
    return np.where(np.isfinite(h_ph), comb, h_ml)                         # no physics -> learning model


def verdict(h_ml, h_ph):
    s_ml, s_ph = state(h_ml), state(h_ph)
    if s_ph == "n/a":
        return {"ok": "🟢 Healthy", "watch": "🟡 Check", "bad": "🔴 Alarm"}.get(s_ml, "– No data") + " (learning model only)"
    worried = [s for s in (s_ml, s_ph) if s in ("watch", "bad")]
    if not worried:
        return "🟢 Healthy"
    if len(worried) == 2:
        return "🔴 Alarm – both models agree" if "bad" in worried else "🟡 Check – both see a change"
    return "🟡 Check – one model only"


def explain(part, h_ml, h_ph, ws, low_load=False):
    """Plain-language reason for the verdict (option 2: physics explains the learning model)."""
    s_ml, s_ph = state(h_ml), state(h_ph)
    if part not in PHYS_PARTS:
        return "No physics model for this part – the learning model decides alone."
    if low_load:
        return ("The turbine has hardly run in the last day, so the parts are not heating up and the physics "
                "check cannot judge them now – the learning model decides alone.")
    if s_ph == "n/a" or ws is None:
        return "No temperature data right now – the learning model decides alone."
    t, gap, band, _ = ws
    name = TARGETS[t][0]
    txt = f"{name[0].upper() + name[1:]} is {gap:+.1f} °C vs what physics expects at this load (normal ±{band:.1f} °C)."
    ml_bad, ph_bad = s_ml in ("watch", "bad"), s_ph in ("watch", "bad")
    if ml_bad and ph_bad:
        return txt + " Both models agree → real heat problem, likely extra friction or losses. Inspect soon."
    if ml_bad and not ph_bad:
        return txt + " Temperature is normal for this load → the learning model's warning is probably not overheating; check other signals or sensors."
    if ph_bad and not ml_bad:
        return txt + " Running hot for its load, but the learning model sees nothing → slow wear it may have learned as normal. Plan an inspection."
    return txt + " Both models: normal."
