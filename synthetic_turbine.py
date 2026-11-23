"""
Live synthetic wind turbine for the digital twin (fault-injection simulator)
============================================================================
Instead of replaying recorded data, this module GENERATES new 10-minute SCADA data that
looks like a real Wind Farm A turbine, and lets you INJECT faults (gearbox overheating,
bearing damage, transformer, hydraulic, converter cooling, pitch misalignment, power loss).

How the data is made
  * Weather: wind speed (persistent AR(1) process + daily cycle), ambient temperature
    (daily cycle + noise), wind direction (random walk). Scenarios: normal, windy, calm,
    storm, hot day, cold day.
  * Turbine: power, rotor / generator speed, nacelle temperature, pitch and every component
    temperature come from Gradient Boosting models trained on a REAL healthy turbine
    (odd days of its training period), plus noise with the same size and persistence as the
    real model errors.
  * Twin: the turbine is monitored with SEPARATE models trained on the EVEN days, exactly like
    the real twin - so the twin never sees the generator's own model.
  * Faults: added on top of the healthy signals, growing suddenly / over 1 day / over 3 days,
    temperature faults scale with load (a damaged bearing heats more at full power).

IMPORTANT: synthetic data shows how the twin REACTS (demo, testing, operator training).
It must NOT be used to claim detection accuracy - real results come from the recorded data.

Command line (exports a scored CSV):
    python synthetic_turbine.py --base 24 --days 10 --fault "Gearbox overheating" --start-day 6
The dashboard uses it through the sidebar: Data source -> Live synthetic turbine.
"""
import os
# hide harmless sklearn/joblib version warnings, also in the parallel worker processes
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import glob

import numpy as np
import pandas as pd

import fault_detection as fd

STEPS_PER_HOUR = 6
RAW_SHOW = {fd.AMBIENT: "ambient_temp", fd.ROTOR_RPM: "rotor_rpm", fd.GEN_RPM: "gen_rpm", fd.NACELLE: "nacelle_temp"}

SCENARIOS = {                      # wind mean factor, extra wind, ambient offset (deg C)
    "Normal weather": (1.0, 0.0, 0.0),
    "Windy": (1.45, 0.0, -2.0),
    "Calm": (0.55, 0.0, 2.0),
    "Storm (very high wind)": (1.0, 9.0, -4.0),
    "Hot day": (0.9, 0.0, 12.0),
    "Cold day": (1.1, 0.0, -12.0),
}
# fault -> component, {signal: weight}, unit of the size
FAULTS = {
    "Gearbox overheating (bearing wear)": ("Gearbox", {"sensor_11_avg": 1.0, "sensor_12_avg": 0.35}, "°C"),
    "Generator bearing damage": ("Generator bearings", {"sensor_13_avg": 1.0, "sensor_14_avg": 0.25}, "°C"),
    "Generator winding overheating": ("Generator windings", {"sensor_15_avg": 1.0, "sensor_16_avg": 1.0, "sensor_17_avg": 1.0}, "°C"),
    "Transformer phase overheating": ("Transformer", {"sensor_39_avg": 1.0, "sensor_38_avg": 0.2}, "°C"),
    "Hydraulic group fault (oil overheating)": ("Hydraulic group", {"sensor_41_avg": 1.0, "pitch_dev": 0.12}, "°C"),
    "Converter cooling failure": ("Converter", {"sensor_35_avg": 1.0, "sensor_36_avg": 1.0, "sensor_37_avg": 1.0, "sensor_21_avg": 0.6}, "°C"),
    "Pitch misalignment": ("Hub / pitch", {"pitch_dev": 1.0, "power_loss": 0.02}, "°"),
    "Blade soiling / icing (power loss)": ("Power performance", {"power_loss": 1.0}, "%"),
}
# temperature / pitch faults: size in multiples of that signal's normal noise (sigma), so the same
# severity means the same thing for every sensor; power loss: fraction of the power.
SEVERITY = {"Mild": {"sigma": 5.0, "%": 0.05},          # below the alarm level -> trend warning
            "Moderate": {"sigma": 17.0, "%": 0.15},     # alarm after a while (sooner at high load)
            "Severe": {"sigma": 25.0, "%": 0.25}}       # alarm within hours
DEVELOPMENT = {"Sudden": 0, "Grows over 1 day": 24, "Grows over 3 days": 72}


# ------------------------------------------------------------------ training (done once per base turbine)
def find_dataset(ds, farm_dir=None):
    here = os.path.dirname(os.path.abspath(__file__))
    farm = farm_dir or fd.find_farm_dir(here)
    if not farm:
        raise FileNotFoundError("Wind Farm A folder not found")
    hits = [f for f in glob.glob(os.path.join(farm, "**", f"{ds}.csv"), recursive=True)]
    if not hits:
        raise FileNotFoundError(f"dataset {ds}.csv not found in {farm}")
    return hits[0]


def _ar1(resid):
    """Noise model (AR(1)) of a signal's error. Robust: unlabelled curtailment / stops in the real
    data would otherwise make the synthetic noise far too heavy - errors beyond 3 robust sigma are
    clipped before the size and persistence are measured."""
    r = pd.Series(resid).dropna()
    if len(r) < 50:
        return 0.0, float(r.std() or 1.0)
    med = r.median()
    rob = 1.4826 * (r - med).abs().median() or r.std() or 1.0
    r = r.clip(med - 3 * rob, med + 3 * rob)
    phi = float(np.clip(r.autocorr(1), 0, 0.995)) if r.std() > 0 else 0.0
    return phi, float(r.std() * np.sqrt(1 - phi ** 2))


def train_models(ds, farm_dir=None, max_rows=6000, seed=0):
    """Generator models (odd days) + twin models (even days) for every monitored signal."""
    path = find_dataset(ds, farm_dir)
    header = fd.read_csv(path, nrows=0).columns
    meta = ["time_stamp", "id", "train_test", "status_type_id"]
    df = fd.read_csv(path, usecols=lambda c: c in meta or c.endswith("_avg"))
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = fd.add_features(df.sort_values("time_stamp").reset_index(drop=True))
    healthy = df[(df["train_test"].astype(str).str.lower() == "train") & df["status_type_id"].isin(fd.NORMAL_STATUS)]
    odd = (healthy["time_stamp"].dt.dayofyear % 2 == 1).to_numpy()
    gen_rows, twin_rows = healthy[odd], healthy[~odd]
    rng = np.random.default_rng(seed)

    def sub(d):
        return d if len(d) <= max_rows else d.iloc[np.sort(rng.choice(len(d), max_rows, replace=False))]

    P, W = fd.POWER, fd.WIND
    p_max = float(healthy[P].quantile(0.999))
    run = healthy[healthy[P] > 0.05 * p_max]
    cut_out = float(min(run[W].quantile(0.999) + 1.0, 25.0)) if len(run) else 25.0
    targets = [s for sens in fd.SUBSYSTEMS.values() for s in sens
               if s in df and s not in ("current_imbalance", "voltage_imbalance", "ae_error")]
    aux = {fd.ROTOR_RPM: [W, P, "power_mean_1h"], fd.GEN_RPM: [W, P, "power_mean_1h"]}
    if fd.NACELLE in df:
        aux[fd.NACELLE] = [c for c in fd.inputs_for(fd.NACELLE, df.columns)]
    m = {"gen": {}, "twin": {}, "noise": {}, "inputs": {}, "calib": {}}
    for t in list(aux) + targets:
        if t not in df or t in m["inputs"]:
            continue
        X_cols = aux.get(t) or fd.inputs_for(t, df.columns)
        X_cols = [c for c in X_cols if c in df and c != t]
        g, w = sub(gen_rows.dropna(subset=[t])), sub(twin_rows.dropna(subset=[t]))
        if len(g) < 300 or len(w) < 300 or g[t].std() < 1e-9:
            continue
        mg = fd.make_model().fit(g[X_cols], g[t])
        mt = fd.make_model().fit(w[X_cols], w[t])
        # noise of the synthetic signal = generator error on unseen (even) days -> realistic size & persistence
        m["noise"][t] = _ar1(w[t] - mg.predict(w[X_cols]))
        # twin calibration = twin error on unseen (odd) days, as in fault_detection.py
        r = g[t] - mt.predict(g[X_cols])
        m["calib"][t] = (float(r.mean()), float(r.std() + 1e-9))
        m["gen"][t], m["twin"][t], m["inputs"][t] = mg, mt, X_cols
    m.update(targets=[t for t in targets if t in m["gen"]], p_max=p_max, cut_out=cut_out,
             wind_mean=float(healthy[W].mean()), wind_std=float(healthy[W].std()),
             amb_mean=float(healthy[fd.AMBIENT].mean()) if fd.AMBIENT in healthy else 15.0,
             base=ds, start=healthy["time_stamp"].max())
    return m


# ------------------------------------------------------------------ the virtual turbine
class SyntheticTurbine:
    def __init__(self, models, scenario="Normal weather", threshold=9.0, crit_limit=18, smooth_hours=6,
                 power_model=None, seed=1, warmup_days=3):
        self.m, self.scenario, self.threshold, self.crit_limit = models, scenario, threshold, crit_limit
        self.win = int(smooth_hours * STEPS_PER_HOUR)
        self.pm = power_model
        self.rng = np.random.default_rng(seed)
        self.t = pd.Timestamp(models["start"]).floor("10min") + pd.Timedelta(days=1)
        self.raw = pd.DataFrame()
        self.z = pd.DataFrame()
        self.out = pd.DataFrame()
        self.state = {"wind": models["wind_mean"], "amb_noise": 0.0, "dir": 200.0, "crit": 0}
        self.noise_state = {k: 0.0 for k in models["noise"]}
        self.faults = []                        # dicts: name, component, start, ramp_h, size, weights, unit
        self.step_id = 0
        if warmup_days:
            self.step(int(warmup_days * 24 * STEPS_PER_HOUR))

    # ---------------- faults
    def inject(self, name, severity="Moderate", development="Grows over 1 day"):
        comp, weights, unit = FAULTS[name]
        size = SEVERITY[severity]["%"] if unit == "%" else SEVERITY[severity]["sigma"]
        main = next((k for k in weights if k in self.m["calib"]), None)
        equiv = (f"{100 * size:.0f} % less power" if unit == "%" else
                 f"+{size * self.m['calib'][main][1]:.1f} {unit} on {fd.SENSOR_NAMES.get(main, main)} at full load" if main else "")
        self.faults.append(dict(name=name, component=comp, weights=weights, unit=unit, severity=severity,
                                size=size, ramp_h=DEVELOPMENT[development], equivalent=equiv,
                                start=self.t, development=development))

    def clear_faults(self):
        for f in self.faults:
            f["end"] = self.t
        self.faults = []

    def _fault_level(self, f, ts):
        if ts < f["start"]:
            return 0.0
        h = (ts - f["start"]).total_seconds() / 3600
        return 1.0 if f["ramp_h"] == 0 else float(min(1.0, h / f["ramp_h"]))

    # ---------------- weather
    def _weather(self, n):
        fac, extra, amb_off = SCENARIOS.get(self.scenario, SCENARIOS["Normal weather"])
        mu = self.m["wind_mean"] * fac + extra
        sd = max(self.m["wind_std"] * (1.2 if extra else 1.0), 0.5)
        phi = 0.985
        rows = []
        for i in range(n):
            ts = self.t + pd.Timedelta(minutes=10 * i)
            hour = ts.hour + ts.minute / 60
            daily = 0.6 * np.sin(2 * np.pi * (hour - 9) / 24)
            w = self.state["wind"]
            w = mu + phi * (w - mu) + self.rng.normal(0, sd * np.sqrt(1 - phi ** 2))
            self.state["wind"] = w
            self.state["amb_noise"] = 0.97 * self.state["amb_noise"] + self.rng.normal(0, 0.25)
            amb = self.m["amb_mean"] + amb_off + 4.0 * np.sin(2 * np.pi * (hour - 9) / 24) + self.state["amb_noise"]
            self.state["dir"] = (self.state["dir"] + self.rng.normal(0, 3)) % 360
            rows.append({"time_stamp": ts, fd.WIND: max(0.0, w + daily), fd.AMBIENT: amb, fd.WIND_DIR: self.state["dir"]})
        return pd.DataFrame(rows)

    def _noise(self, key, n):
        phi, s = self.m["noise"][key]
        out = np.empty(n)
        e = self.noise_state.get(key, 0.0)
        for i in range(n):
            e = phi * e + self.rng.normal(0, s)
            out[i] = e
        self.noise_state[key] = e
        return out

    def _gen(self, key, frame, idx):
        X = frame.loc[idx, self.m["inputs"][key]]
        return self.m["gen"][key].predict(X) + self._noise(key, len(idx))

    # ---------------- one batch of n new 10-minute rows
    def step(self, n=6):
        P, W = fd.POWER, fd.WIND
        new = self._weather(n)
        hist = self.raw.tail(80)
        frame = pd.concat([hist, new], ignore_index=True)
        idx = frame.index[len(hist):]
        # weather features needed by the power model
        frame["wind_cubed"] = frame[W].clip(lower=0) ** 3
        frame["wind_std_1h"] = frame[W].rolling(STEPS_PER_HOUR, min_periods=2).std()
        r = np.deg2rad(frame[fd.WIND_DIR] % 360)
        frame["dir_sin"], frame["dir_cos"] = np.sin(r), np.cos(r)
        # power: model + noise, limits, cut-out, power-loss faults
        p = self._gen(P, frame, idx) if P in self.m["gen"] else np.zeros(n)
        loss = np.zeros(n)
        for f in self.faults:
            if "power_loss" in f["weights"]:
                size = f["size"] if f["unit"] == "%" else f["weights"]["power_loss"] * f["size"] / 10
                loss += np.array([self._fault_level(f, ts) for ts in frame.loc[idx, "time_stamp"]]) * size
        p = np.clip(p, 0, self.m["p_max"] * 1.01) * (1 - np.clip(loss, 0, 0.9))
        p[frame.loc[idx, W].to_numpy() > self.m["cut_out"]] = 0.0
        frame.loc[idx, P] = p
        for k in (fd.ROTOR_RPM, fd.GEN_RPM):
            if k in self.m["gen"]:
                frame.loc[idx, "power_mean_1h"] = frame[P].rolling(STEPS_PER_HOUR, min_periods=1).mean().loc[idx]
                frame.loc[idx, k] = np.clip(self._gen(k, frame, idx), 0, None) * (p > 0.005 * self.m["p_max"])
        frame = fd.add_features(frame)
        if fd.NACELLE in self.m["gen"]:
            frame.loc[idx, fd.NACELLE] = self._gen(fd.NACELLE, frame, idx)
        load = np.clip(frame.loc[idx, "power_mean_1h"].to_numpy() / max(self.m["p_max"], 1e-9), 0, 1)
        for t in self.m["targets"]:
            if t == P:
                continue
            v = self._gen(t, frame, idx)
            for f in self.faults:
                wgt = f["weights"].get(t)
                if wgt:
                    lvl = np.array([self._fault_level(f, ts) for ts in frame.loc[idx, "time_stamp"]])
                    scale = (0.6 + 0.4 * load) if f["unit"] == "°C" else 1.0       # heat grows with load
                    sd = self.m["calib"].get(t, (0, 1.0))[1]
                    v = v + wgt * f["size"] * sd * lvl * scale
            frame.loc[idx, t] = v
        if "pitch_dev" in frame:
            frame.loc[idx, "sensor_5_avg"] = frame.loc[idx, "pitch_dev"] % 360
        batch = frame.loc[idx].copy()
        batch["id"] = np.arange(self.step_id, self.step_id + n)
        batch["fault_label"] = " + ".join(f["name"] for f in self.faults) if self.faults else ""
        self.step_id += n
        self.t = self.t + pd.Timedelta(minutes=10 * n)
        self._score(batch)
        self.raw = pd.concat([self.raw, batch], ignore_index=True).tail(6 * 24 * STEPS_PER_HOUR)
        return self

    # ---------------- the twin: same scoring rules as fault_detection.py
    def _score(self, batch):
        P = fd.POWER
        z = pd.DataFrame(index=batch.index)
        for t in self.m["targets"]:
            pred = self.m["twin"][t].predict(batch[self.m["inputs"][t]])
            mu, sd = self.m["calib"][t]
            zt = (batch[t].to_numpy() - pred - mu) / sd
            if t == P:
                zt = -zt
                zt[batch[P].to_numpy() <= 0.02 * self.m["p_max"]] = np.nan
            if t in fd.TWO_SIDED:
                zt = np.abs(zt)
            z[f"z_{t}"] = zt
        self.z = pd.concat([self.z, z], ignore_index=True).tail(30 * 24 * STEPS_PER_HOUR).reset_index(drop=True)
        zs = self.z.tail(len(z) + self.win).rolling(self.win, min_periods=max(1, self.win // 3)).mean().clip(lower=0).tail(len(z))
        sub = pd.DataFrame(index=zs.index)
        for name, sens in fd.SUBSYSTEMS.items():
            cols = [f"z_{s}" for s in sens if f"z_{s}" in zs]
            if cols:
                sub[name] = zs[cols].max(axis=1).to_numpy()
        score = sub.max(axis=1).to_numpy()
        crit, alarm = [], []
        c = self.state["crit"]
        for s in score:
            c = c + 1 if (not np.isnan(s) and s > self.threshold) else max(c - 1, 0)
            crit.append(c); alarm.append(c >= self.crit_limit)
        self.state["crit"] = c
        out = pd.DataFrame({"time_stamp": batch["time_stamp"].to_numpy(), "id": batch["id"].to_numpy(),
                            "status_type_id": 0, "monitored": True})
        out = pd.concat([out, sub.reset_index(drop=True).round(3), z.reset_index(drop=True).round(3)], axis=1)
        out["anomaly_score"] = np.round(score, 3)
        out["threshold"] = self.threshold
        out["criticality"] = crit
        out["alarm"] = alarm
        out["in_event_window"] = (batch["fault_label"] != "").to_numpy()
        out["fault_label"] = batch["fault_label"].to_numpy()
        out["wind"] = batch[fd.WIND].to_numpy()
        out["power"] = batch[P].to_numpy()
        for src, dst in RAW_SHOW.items():
            out[dst] = batch[src].to_numpy() if src in batch else np.nan
        out["pitch"] = batch["pitch_dev"].to_numpy() if "pitch_dev" in batch else np.nan
        out["pred_ml"], out["pred_phys"] = self._expected_power(batch)
        self.out = pd.concat([self.out, out], ignore_index=True).tail(30 * 24 * STEPS_PER_HOUR).reset_index(drop=True)

    def _expected_power(self, batch):
        """Expected power from the trained power model (power_model.py), if available."""
        if not self.pm:
            return np.full(len(batch), np.nan), np.full(len(batch), np.nan)
        R, PSTD, RHO = 287.05, 101325.0, 1.225
        w = batch[fd.WIND].to_numpy()
        t = batch[fd.AMBIENT].to_numpy()
        rho = PSTD / (R * (t + 273.15))
        ts = pd.to_datetime(batch["time_stamp"])
        hour = (ts.dt.hour + ts.dt.minute / 60).to_numpy()
        doy = ts.dt.dayofyear.to_numpy()
        prev = self.raw[fd.WIND].tail(1).to_numpy() if not self.raw.empty else np.array([])
        f = {"wind": w, "wind2": w ** 2, "wind3": w ** 3, "wind_norm": w * (rho / RHO) ** (1 / 3),
             "wind_std_1h": batch["wind_std_1h"].fillna(0.5).to_numpy(),
             "wind_change": np.diff(np.concatenate([prev if len(prev) else w[:1], w])),
             "air_density": rho, "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
             "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25), "temp": t,
             "dir_sin": batch["dir_sin"].to_numpy(), "dir_cos": batch["dir_cos"].to_numpy()}
        X = pd.DataFrame({c: f[c] for c in self.pm["features"]})
        ml = np.clip(self.pm["model"].predict(X), 0, self.pm["rated_power"] * 1.02)
        phys = self.pm["power_curve"].predict(f["wind_norm"])
        return ml, phys

    def frame(self):
        return self.out


# ------------------------------------------------------------------ command line: export a scored CSV
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=int, required=True, help="healthy dataset id to imitate (e.g. a normal dataset)")
    ap.add_argument("--days", type=float, default=10)
    ap.add_argument("--scenario", default="Normal weather", choices=list(SCENARIOS))
    ap.add_argument("--fault", default=None, help="fault name (start of the name is enough), e.g. 'Gearbox'")
    ap.add_argument("--severity", default="Moderate", choices=list(SEVERITY))
    ap.add_argument("--development", default="Grows over 1 day", choices=list(DEVELOPMENT))
    ap.add_argument("--start-day", type=float, default=5, help="day on which the fault starts")
    ap.add_argument("--threshold", type=float, default=9.0)
    ap.add_argument("--crit-limit", type=int, default=18)
    ap.add_argument("--out", default="synthetic_turbine.csv")
    a = ap.parse_args()
    print(f"Training generator + twin models on dataset {a.base} ...")
    m = train_models(a.base)
    tb = SyntheticTurbine(m, a.scenario, a.threshold, a.crit_limit, warmup_days=0)
    total = int(a.days * 24 * STEPS_PER_HOUR)
    fault_at = int(a.start_day * 24 * STEPS_PER_HOUR)
    name = next((k for k in FAULTS if a.fault and k.lower().startswith(a.fault.lower())), None)
    if a.fault and not name:
        raise SystemExit(f"Unknown fault. Choose from: {list(FAULTS)}")
    done = 0
    while done < total:
        if name and done == fault_at:
            tb.inject(name, a.severity, a.development)
            print(f"day {done / 144:.1f}: injected '{name}' ({a.severity}, {a.development})")
        n = min(36, total - done, (fault_at - done) if name and done < fault_at else 36)
        tb.step(n); done += n
    df = tb.frame()
    df.to_csv(a.out, index=False)
    al = df[df["alarm"].astype(bool)]
    print(f"Saved {len(df)} rows to {a.out}")
    if name:
        first = al[al["in_event_window"]]["time_stamp"].min() if len(al) else None
        print("First alarm:", first, "->", (pd.Timestamp(first) - df.loc[df["in_event_window"], "time_stamp"].min())
              if first is not None else "no alarm")
    print("False alarms before the fault:", int(al[~al["in_event_window"]].shape[0]), "steps")


if __name__ == "__main__":
    main()
