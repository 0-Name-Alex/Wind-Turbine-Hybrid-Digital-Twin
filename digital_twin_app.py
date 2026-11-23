"""
Wind Turbine Digital Twin - dashboard (Part 3)
==============================================
Combines the outputs of
    power_model.py      -> outputs_power_model/
    fault_detection.py  -> outputs_fault_detection/
(and reads a few raw SCADA signals from the Wind Farm A folder)
into one interactive digital twin.

Run (in the same folder as the two scripts):
    conda install -c conda-forge streamlit plotly -y
    streamlit run digital_twin_app.py

Pages
    Overview          - live KPIs, power vs twin + 3-h forecast, insights, anomalies, health, alerts
    Power Prediction  - model accuracy, forecast skill, power curve, what-if
    Anomaly Detection - subsystem scores, recommended actions (do now / prevent), anomaly log, results
    Turbine Health    - turbine view, health table with trend and time-to-alarm
    Hybrid Twin       - learning model + physics model side by side, combined verdict and reason [physics_twin.py]
    3D Twin           - 3D turbine (rotor speed, pitch, wind from the data; parts coloured by health) [twin3d.py]
    Reports           - all results for the report

Data sources (sidebar)
    Recorded data (replay)  - the real test period of each dataset, replayed step by step
    Live synthetic turbine  - synthetic_turbine.py generates new data; faults can be injected

Optional extra outputs used if present:
    forecast_model.py -> outputs_forecast/   (short-term power forecast)
    health_trend.py   -> time-to-alarm evaluation (Reports page)
"""

import glob
import json
import os
# hide harmless sklearn/joblib version warnings, also in the parallel worker processes
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

try:                                     # shared feature pipeline + health trend
    import fault_detection as fd
except Exception:                        # dashboard still works without it
    fd = None
try:                                     # live synthetic turbine (fault-injection simulator)
    import synthetic_turbine as syn
except Exception:
    syn = None
try:                                     # 3D view (self-contained WebGL, works offline)
    import twin3d
except Exception:
    twin3d = None
try:                                     # physics model (same equations as the MATLAB twin)
    import physics_twin as pt
except Exception:
    pt = None
SYN = "SYN"

HERE = os.path.dirname(os.path.abspath(__file__))
POWER_DIR = os.path.join(HERE, "outputs_power_model")
FAULT_DIR = os.path.join(HERE, "outputs_fault_detection")
FORECAST_DIR = os.path.join(HERE, "outputs_forecast")
HORIZONS = {1: "10 min", 6: "1 h", 12: "2 h", 18: "3 h"}

R_DRY_AIR, P_STD, RHO_REF = 287.05, 101325.0, 1.225
STEPS_PER_HOUR = 6
FRAME_SECONDS = 0.6                 # replay: time between frames

# ---------------------------------------------------------------- theme
TEAL, TEAL_LIGHT = "#0E9F8E", "#DDF4F0"
BLUE, BLUE_LIGHT = "#2F80ED", "#DCEAFD"
INK, MUTED, LINE = "#1F2D3D", "#6B7C93", "#E6EEF5"
GREEN, AMBER, RED, GREY, PURPLE = "#16A34A", "#F59E0B", "#E5484D", "#A3B1C2", "#8B5CF6"
FONT = "Inter, 'Segoe UI', Roboto, Arial, sans-serif"

SUBSYSTEMS = ["Generator bearings", "Gearbox", "Generator windings", "Transformer",
              "Hydraulic group", "Hub / pitch", "Converter", "Nacelle cabinets",
              "Electrical balance", "Power performance", "Whole turbine (AE)"]
PARTS = {
    "hub":         ["Hub / pitch", "Hydraulic group"],
    "blades":      ["Power performance"],
    "gearbox":     ["Gearbox"],
    "generator":   ["Generator bearings", "Generator windings"],
    "converter":   ["Converter", "Nacelle cabinets", "Electrical balance"],
    "transformer": ["Transformer"],
    "nacelle":     ["Whole turbine (AE)"],
}
RAW_COLS = {"sensor_0_avg": "ambient_temp", "sensor_52_avg": "rotor_rpm", "sensor_18_avg": "gen_rpm",
            "sensor_43_avg": "nacelle_temp", "sensor_5_avg": "pitch"}

# ---------------------------------------------------------------- pickled class
class BinnedPowerCurve:
    def __init__(self, bin_width=0.5):
        self.bin_width = bin_width

    def predict(self, wind_norm):
        return np.interp(wind_norm, self.v_, self.p_, left=0.0, right=self.p_[-1])


def _register_pickle_classes():
    import __main__
    if not hasattr(__main__, "BinnedPowerCurve"):
        __main__.BinnedPowerCurve = BinnedPowerCurve


# ================================================================ pure helpers
def fmt(x, nd=1, suffix=""):
    if x is None or pd.isna(x):
        return "–"
    x = 0.0 if round(float(x), nd) == 0 else float(x)     # avoid "-0"
    return f"{x:.{nd}f}{suffix}"


def health_from_score(score, threshold):
    """0-100 %: 100 = perfectly normal, 0 = at/over the alarm threshold."""
    if score is None or pd.isna(score) or not threshold:
        return np.nan
    return float(np.clip(100.0 * (1.0 - score / threshold), 0, 100))


def colour_for(health):
    if health is None or pd.isna(health):
        return GREY
    return GREEN if health >= 70 else AMBER if health >= 30 else RED


def turbine_status(row):
    thr = row.get("threshold", np.nan)
    score = row.get("anomaly_score", np.nan)
    if bool(row.get("alarm", False)):
        return "Alarm", RED
    if (not pd.isna(score) and thr and score > 0.5 * thr) or row.get("criticality", 0) > 0:
        return "Warning", AMBER
    if pd.isna(score):
        return "No data", GREY
    return "Normal", GREEN


def weather_features(wind, temp_c, direction_deg, turbulence, hour, day_of_year, features):
    """Build the exact input row the power model was trained on."""
    rho = P_STD / (R_DRY_AIR * (temp_c + 273.15))
    f = {
        "wind": wind, "wind2": wind ** 2, "wind3": wind ** 3,
        "wind_norm": wind * (rho / RHO_REF) ** (1 / 3),
        "wind_std_1h": turbulence, "wind_change": 0.0, "air_density": rho,
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25), "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "temp": temp_c,
        "dir_sin": np.sin(np.deg2rad(direction_deg)), "dir_cos": np.cos(np.deg2rad(direction_deg)),
    }
    return pd.DataFrame([[f[c] for c in features]], columns=features)


def episodes(mask, min_len=1):
    """(start, end) index pairs of consecutive True runs, at least min_len long."""
    m = np.asarray(mask, dtype=bool)
    out, i, n = [], 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def event_list(so_far, sub_cols, sigma):
    """All alarms, warnings and power deviations up to twin time, newest first."""
    ev = []
    thr = float(so_far["threshold"].iloc[-1]) if len(so_far) else np.nan
    t = so_far["time_stamp"].reset_index(drop=True)
    sub = so_far[sub_cols].reset_index(drop=True)
    score = so_far["anomaly_score"].reset_index(drop=True)
    alarm = so_far["alarm"].fillna(False).astype(bool).to_numpy()

    def peak_sub(s, e):
        seg = sub.iloc[s:e + 1].max()
        return (seg.idxmax(), float(seg.max())) if seg.notna().any() else ("-", np.nan)

    for s, e in episodes(alarm):
        name, pk = peak_sub(s, e)
        ev.append({"time": t[s], "type": f"{name} anomaly", "severity": "High",
                   "details": f"Alarm · peak score {pk:.1f} (threshold {thr:.1f}) · {(e - s + 1) / 6:.1f} h"})
    warn = ((score > 0.6 * thr) & ~pd.Series(alarm)).to_numpy() if thr == thr else np.zeros(len(t), bool)
    for s, e in episodes(warn, min_len=STEPS_PER_HOUR):
        name, pk = peak_sub(s, e)
        ev.append({"time": t[s], "type": f"{name} deviation", "severity": "Medium",
                   "details": f"Score up to {pk:.1f} ({100 * pk / thr:.0f} % of threshold) · {(e - s + 1) / 6:.1f} h"})
    if sigma and {"power", "pred_ml"} <= set(so_far.columns):
        p, x = so_far["power"].to_numpy(), so_far["pred_ml"].to_numpy()
        low = (x > 0.1) & (p < x - 2.5 * sigma)
        for s, e in episodes(low, min_len=STEPS_PER_HOUR):
            gap = float(np.nanmean(x[s:e + 1] - p[s:e + 1])) * 100
            ev.append({"time": t[s], "type": "Power deviation", "severity": "High" if gap > 15 else "Low",
                       "details": f"Output {gap:.0f} % of rated below the twin for {(e - s + 1) / 6:.1f} h"})
    return sorted(ev, key=lambda d: d["time"], reverse=True)


# ---------------------------------------------------------------- HTML pieces
BASE_CSS = ""          # kept so older helper code that prepends it still works
CARD_CSS = f"""
.tw{{font-family:{FONT};color:{INK};margin-bottom:6px}} .tw *{{box-sizing:border-box}}
.tw .card{{background:#fff;border-radius:16px;border:1px solid {LINE};box-shadow:0 6px 20px rgba(30,80,120,.07);padding:16px 18px}}
.tw .h{{display:flex;align-items:center;gap:10px;font-weight:700;font-size:16px;margin-bottom:10px}}
.tw .ic{{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:16px}}
.tw .muted{{color:{MUTED};font-size:12.5px}}
.tw .pill{{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}}
.tw .High{{background:#FDE8E8;color:#C81E1E}} .tw .Medium{{background:#FEF3C7;color:#B45309}}
.tw .Low{{background:#FEF9C3;color:#A16207}} .tw .Info{{background:{BLUE_LIGHT};color:#1D4ED8}}
.tw p{{margin:0}}
"""


def show_html(html, height=None):
    """Render an HTML card inline (no iframe), so it updates in place without flashing during replay.
    Written on one line so Markdown leaves it as raw HTML."""
    one_line = " ".join(line.strip() for line in html.replace(BASE_CSS, "").splitlines() if line.strip())
    st.markdown(f'<div class="tw">{one_line}</div>', unsafe_allow_html=True)


def sparkline_svg(values, colour, w=150, h=38):
    v = np.asarray([x for x in values if x == x], dtype=float)
    if len(v) < 2:
        return f'<svg width="{w}" height="{h}"></svg>'
    lo, hi = v.min(), v.max()
    rng = hi - lo if hi > lo else 1.0
    xs = np.linspace(2, w - 2, len(v))
    ys = h - 4 - (v - lo) / rng * (h - 8)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    area = f"2,{h} " + pts + f" {w - 2},{h}"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polygon points="{area}" fill="{colour}" fill-opacity="0.10"/>'
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"/></svg>')


def delta_html(now, before, unit="%", absolute=False):
    if now is None or before is None or pd.isna(now) or pd.isna(before):
        return '<span class="muted">no change data</span>'
    if absolute or abs(before) < 1e-6:
        d, txt = now - before, f"{abs(now - before):.1f}{unit}"
    else:
        d = (now - before) / abs(before) * 100
        txt = f"{abs(d):.1f}%"
    arrow, col = ("↑", GREEN) if d > 0 else ("↓", RED) if d < 0 else ("→", MUTED)
    return f'<span style="color:{col};font-weight:700">{arrow} {txt}</span> <span class="muted">vs last hour</span>'


def kpi_cards_html(cards):
    """cards: list of dict(icon, icon_bg, title, value, delta, spark|None, big_colour|None, sub|None)."""
    items = []
    for c in cards:
        body = (f'<div style="font-size:30px;font-weight:700;margin:6px 0 2px;color:{c.get("big_colour", INK)}">{c["value"]}</div>'
                f'<div style="font-size:12.5px">{c["delta"]}</div>')
        spark = f'<div style="margin-top:8px">{c["spark"]}</div>' if c.get("spark") else ""
        items.append(f'<div class="card" style="flex:1;min-width:0">'
                     f'<div class="h" style="font-size:14px;font-weight:600"><div class="ic" style="background:{c["icon_bg"]}">{c["icon"]}</div>{c["title"]}</div>'
                     f'{body}{spark}</div>')
    return BASE_CSS + f'<div style="display:flex;gap:16px">{"".join(items)}</div>'


def hero_html(dataset_text, twin_time, step, n, extra="", mode="Replay mode", dot=None):
    turb = ('<g stroke="#ffffff" stroke-linecap="round">'
            '<line x1="0" y1="0" x2="0" y2="150" stroke-width="7"/>'
            '<g><line x1="0" y1="0" x2="0" y2="-78" stroke-width="7"/><line x1="0" y1="0" x2="68" y2="39" stroke-width="7"/>'
            '<line x1="0" y1="0" x2="-68" y2="39" stroke-width="7"/>'
            '<animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="9s" repeatCount="indefinite"/></g>'
            '<circle r="7" fill="#ffffff"/></g>')
    hills = ('<path d="M0 175 C 120 140, 220 160, 330 150 S 560 125, 700 150 L 700 200 L 0 200 Z" fill="#cfe6f3"/>'
             '<path d="M0 185 C 150 165, 300 180, 450 168 S 620 160, 700 172 L 700 200 L 0 200 Z" fill="#b9dcee"/>')
    return BASE_CSS + f"""
<div style="position:relative;height:178px;border-radius:18px;overflow:hidden;
     background:linear-gradient(120deg,#ffffff 0%,#e9f5fc 45%,#cfe8f6 100%);border:1px solid {LINE}">
  <svg style="position:absolute;right:0;bottom:0" width="700" height="200" viewBox="0 0 700 200">{hills}
    <g transform="translate(575,92) scale(0.85)">{turb}</g><g transform="translate(430,118) scale(0.5)" opacity="0.8">{turb}</g></svg>
  <div style="position:absolute;left:26px;top:22px">
    <div style="letter-spacing:.14em;font-size:11px;font-weight:700;color:{MUTED}">SCADA-DRIVEN MONITORING &amp; DIAGNOSIS</div>
    <div style="font-size:30px;font-weight:800;margin:6px 0 8px;color:#0B3B5C">Wind Turbine Digital Twin</div>
    <div style="font-size:13.5px;color:#2C4A63">Power prediction &nbsp;•&nbsp; Anomaly detection &nbsp;•&nbsp; Turbine health</div>
    <div style="margin-top:14px;font-size:13px;color:#2C4A63">{dataset_text}</div>
  </div>
  <div style="position:absolute;right:20px;top:16px;display:flex;gap:10px">
    <span class="pill" style="background:#ffffffd9;color:{INK}"><span style="color:{dot or GREEN}">●</span> {mode}</span>
    <span class="pill" style="background:#ffffffd9;color:{INK}">🗓 {twin_time}</span>
    <span class="pill" style="background:#ffffffd9;color:{MUTED}">step {step} / {n}</span>{extra}
  </div>
</div>"""


def insights_html(items, min_h=330):
    icons = {"ok": ("✓", GREEN, "#DCFCE7"), "warn": ("!", AMBER, "#FEF3C7"), "bad": ("✕", RED, "#FDE8E8"), "info": ("i", BLUE, BLUE_LIGHT)}
    rows = "".join(
        f'<div style="display:flex;gap:10px;align-items:flex-start;margin:9px 0">'
        f'<div style="min-width:22px;height:22px;border-radius:50%;background:{icons[k][2]};color:{icons[k][1]};'
        f'font-weight:800;font-size:12px;display:flex;align-items:center;justify-content:center">{icons[k][0]}</div>'
        f'<div style="font-size:13.5px;line-height:1.35">{t}</div></div>' for k, t in items)
    return BASE_CSS + f'<div class="card" style="min-height:{min_h}px"><div class="h"><div class="ic" style="background:{TEAL_LIGHT}">💡</div>Key insights</div>{rows}</div>'


def health_card_html(overall, weakest, readings, min_h=280):
    pct = 0 if pd.isna(overall) else overall
    col = colour_for(overall)
    r, c = 52, 2 * np.pi * 52
    dash = c * pct / 100
    gauge = (f'<svg width="140" height="140" viewBox="0 0 140 140"><circle cx="70" cy="70" r="{r}" stroke="{LINE}" stroke-width="12" fill="none"/>'
             f'<circle cx="70" cy="70" r="{r}" stroke="{col}" stroke-width="12" fill="none" stroke-linecap="round" '
             f'stroke-dasharray="{dash:.1f} {c:.1f}" transform="rotate(-90 70 70)"/>'
             f'<text x="70" y="70" text-anchor="middle" font-size="26" font-weight="800" fill="{INK}" font-family="{FONT}">{fmt(overall, 0)}%</text>'
             f'<text x="70" y="92" text-anchor="middle" font-size="12" fill="{col}" font-weight="700" font-family="{FONT}">'
             f'{"Healthy" if pct >= 70 else "Watch" if pct >= 30 else "Critical"}</text></svg>')
    rows = "".join(f'<div style="display:flex;justify-content:space-between;gap:10px;padding:6px 0;border-bottom:1px dashed {LINE}">'
                   f'<span class="muted">{ic} {k}</span><b style="font-size:13px">{v}</b></div>' for ic, k, v in readings)
    return BASE_CSS + f"""<div class="card" style="min-height:{min_h}px"><div class="h"><div class="ic" style="background:#FCE7F3">💚</div>Turbine health</div>
<div style="display:flex;gap:16px;align-items:center"><div style="text-align:center">{gauge}
<div class="muted" style="max-width:140px">weakest: {weakest}</div></div><div style="flex:1">{rows}</div></div></div>"""


def alerts_html(events, n=5, min_h=280):
    if not events:
        rows = f'<div class="muted" style="padding:10px 0">No alerts up to twin time.</div>'
    else:
        dot = {"High": RED, "Medium": AMBER, "Low": "#EAB308", "Info": BLUE}
        rows = "".join(
            f'<div style="display:flex;gap:10px;align-items:flex-start;padding:9px 0;border-bottom:1px solid {LINE}">'
            f'<span style="color:{dot[e["severity"]]};font-size:14px;line-height:1.2">●</span>'
            f'<div style="flex:1"><div style="font-size:13.5px;font-weight:600">{e["type"]}</div>'
            f'<div class="muted">{pd.Timestamp(e["time"]):%d %b %H:%M}</div></div>'
            f'<span class="pill {e["severity"]}">{e["severity"]}</span></div>' for e in events[:n])
    return BASE_CSS + f'<div class="card" style="min-height:{min_h}px"><div class="h"><div class="ic" style="background:#FEF3C7">🔔</div>Recent alerts</div>{rows}</div>'


def anomaly_table_html(events, n=4, min_h=330):
    head = "".join(f'<th style="text-align:left;padding:7px 8px;font-size:12px;color:{MUTED};border-bottom:1px solid {LINE}">{h}</th>'
                   for h in ("Time", "Type", "Severity", "Details"))
    if not events:
        body = f'<tr><td colspan="4" class="muted" style="padding:10px 8px">No anomalies up to twin time.</td></tr>'
    else:
        body = "".join(f'<tr><td style="padding:8px;font-size:12.5px;white-space:nowrap">{pd.Timestamp(e["time"]):%d %b %H:%M}</td>'
                       f'<td style="padding:8px;font-size:12.5px">{e["type"]}</td>'
                       f'<td style="padding:8px"><span class="pill {e["severity"]}">{e["severity"]}</span></td>'
                       f'<td style="padding:8px;font-size:12.5px;color:{MUTED}">{e["details"]}</td></tr>' for e in events[:n])
    return BASE_CSS + (f'<div class="card" style="min-height:{min_h}px"><div class="h" style="margin-bottom:4px">Recent anomalies</div>'
                       f'<table style="width:100%;border-collapse:collapse"><tr>{head}</tr>{body}</table></div>')


def metric_cards_html(items):
    cells = "".join(f'<div class="card" style="flex:1;padding:14px 16px"><div class="muted">{k}</div>'
                    f'<div style="font-size:26px;font-weight:800;margin-top:4px">{v}</div>'
                    f'<div class="muted" style="margin-top:2px">{sub}</div></div>' for k, v, sub in items)
    return BASE_CSS + f'<div style="display:flex;gap:14px">{cells}</div>'


def turbine_svg(part_health, power_frac, status, status_colour):
    """Animated turbine drawing; each part coloured by its health."""
    def c(part):
        return colour_for(part_health.get(part, (np.nan, ""))[0])

    def tip(part):
        h, label = part_health.get(part, (np.nan, ""))
        return f"{label}: {'no data' if pd.isna(h) else f'{h:.0f}% health'}"

    spin = "" if (pd.isna(power_frac) or power_frac < 0.02) else (
        f'<animateTransform attributeName="transform" type="rotate" from="0 120 130" '
        f'to="360 120 130" dur="{max(0.7, 5.0 * (1 - min(power_frac, 1.0)) + 0.7):.2f}s" repeatCount="indefinite"/>')
    blade = ('<path d="M120 130 C 112 95, 114 50, 120 22 C 126 50, 128 95, 120 130 Z" '
             f'fill="{c("blades")}" fill-opacity="0.85" stroke="#2b3440" stroke-width="1"/>')
    legend = "".join(
        f'<g transform="translate({12 + i * 112},388)"><rect width="12" height="12" rx="2" fill="{col}"/>'
        f'<text x="17" y="10" font-size="11" fill="{MUTED}">{txt}</text></g>'
        for i, (col, txt) in enumerate([(GREEN, "healthy"), (AMBER, "watch"), (RED, "critical")]))
    return f"""
<div style="font-family:{FONT}; color:{INK};">
<svg viewBox="0 0 360 410" width="100%" style="max-width:430px; display:block; margin:auto">
  <line x1="10" y1="372" x2="350" y2="372" stroke="{GREY}" stroke-width="2"/>
  <path d="M186 152 L200 152 L210 372 L176 372 Z" fill="#E3ECF3" stroke="{GREY}"/>
  <g><title>{tip("transformer")}</title>
    <rect x="222" y="330" width="52" height="42" rx="4" fill="{c("transformer")}" stroke="#2b3440"/>
    <text x="248" y="355" font-size="10" text-anchor="middle" fill="white">TRAFO</text></g>
  <g><title>{tip("blades")}</title>
    <g>{blade}<g transform="rotate(120 120 130)">{blade}</g><g transform="rotate(240 120 130)">{blade}</g>{spin}</g></g>
  <g><title>{tip("nacelle")}</title>
    <rect x="128" y="110" width="196" height="48" rx="10" fill="#F4F8FB" stroke="{c("nacelle")}" stroke-width="4"/></g>
  <g><title>{tip("gearbox")}</title>
    <rect x="142" y="119" width="54" height="30" rx="4" fill="{c("gearbox")}" stroke="#2b3440"/>
    <text x="169" y="137.5" font-size="8.5" font-weight="600" text-anchor="middle" fill="white">GEARBOX</text></g>
  <g><title>{tip("generator")}</title>
    <rect x="201" y="119" width="64" height="30" rx="4" fill="{c("generator")}" stroke="#2b3440"/>
    <text x="233" y="137.5" font-size="8.5" font-weight="600" text-anchor="middle" fill="white">GENERATOR</text></g>
  <g><title>{tip("converter")}</title>
    <rect x="270" y="119" width="46" height="30" rx="4" fill="{c("converter")}" stroke="#2b3440"/>
    <text x="293" y="137.5" font-size="8.5" font-weight="600" text-anchor="middle" fill="white">CONV.</text></g>
  <g><title>{tip("hub")}</title><circle cx="120" cy="130" r="15" fill="{c("hub")}" stroke="#2b3440" stroke-width="1.5"/></g>
  <rect x="236" y="14" width="112" height="30" rx="15" fill="{status_colour}"/>
  <text x="292" y="34" font-size="14" font-weight="700" text-anchor="middle" fill="white">{status.upper()}</text>
  {legend}
</svg>
<div style="text-align:center; font-size:12px; color:{MUTED}">Hover a part for its health. Rotor speed follows power output.</div>
</div>"""


# ================================================================ data loading
def find_farm_dir():
    for base in (HERE, os.path.dirname(HERE)):
        hits = [d for d in glob.glob(os.path.join(base, "**", "Wind Farm A"), recursive=True)
                if os.path.isfile(os.path.join(d, "event_info.csv"))]
        if hits:
            return hits[0]
    return None


@st.cache_resource
def load_power_model():
    path = os.path.join(POWER_DIR, "power_model.joblib")
    if not os.path.isfile(path):
        return None
    _register_pickle_classes()
    return joblib.load(path)


@st.cache_data
def load_csv(path, parse_time=True):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if parse_time and "time_stamp" in df:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    return df


@st.cache_data
def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def power_error_sigma():
    """Std of the ML model error on unseen normal data (fraction of rated)."""
    pw = load_csv(os.path.join(POWER_DIR, "test_predictions.csv"))
    if pw is None:
        return None
    ok = pw["pred_ml"] > 0.02
    if "event_label" in pw:
        ok &= pw["event_label"].astype(str).str.lower() == "normal"
    r = (pw.loc[ok, "power"] - pw.loc[ok, "pred_ml"]).dropna()
    return float(r.std()) if len(r) > 50 else None


@st.cache_data
def dataset_list():
    res = load_csv(os.path.join(FAULT_DIR, "results_per_dataset.csv"), parse_time=False)
    ids = sorted(int(os.path.splitext(os.path.basename(f))[0])
                 for f in glob.glob(os.path.join(FAULT_DIR, "scores", "*.csv")))
    info = {}
    for ds in ids:
        r = res[res["dataset_id"] == ds].iloc[0].to_dict() if res is not None and (res["dataset_id"] == ds).any() else {}
        label = str(r.get("label", "?"))
        fault = "" if pd.isna(r.get("fault", "")) else str(r.get("fault", ""))
        asset = r.get("asset", "?")
        info[ds] = {"asset": asset, "label": label, "fault": fault,
                    "text": f"Dataset {ds} · Asset {asset} · " + (f"Fault: {fault}" if label == "anomaly" else "Normal")}
    return info


@st.cache_data
def load_twin_data(ds):
    """Fault scores + power predictions + a few raw SCADA signals, aligned on time."""
    sc = load_csv(os.path.join(FAULT_DIR, "scores", f"{ds}.csv"))
    pw = load_csv(os.path.join(POWER_DIR, "test_predictions.csv"))
    if pw is not None:
        pw = pw[pw["dataset_id"] == ds][["time_stamp", "wind", "power", "pred_ml", "pred_phys"]]
        sc = sc.merge(pw.drop_duplicates("time_stamp"), on="time_stamp", how="left")
    farm = find_farm_dir()
    raw_path = os.path.join(farm, "datasets", f"{ds}.csv") if farm else None
    if raw_path and os.path.isfile(raw_path):
        with open(raw_path, encoding="utf-8", errors="ignore") as f:
            sep = ";" if f.readline().count(";") > 1 else ","
        raw = pd.read_csv(raw_path, sep=sep, usecols=lambda c: c in ["time_stamp", *RAW_COLS])
        raw["time_stamp"] = pd.to_datetime(raw["time_stamp"])
        raw = raw.rename(columns=RAW_COLS).drop_duplicates("time_stamp")
        if "pitch" in raw:
            raw["pitch"] = ((raw["pitch"] + 180) % 360) - 180
        sc = sc.merge(raw, on="time_stamp", how="left")
    fc = load_csv(os.path.join(FORECAST_DIR, "forecasts.csv"))
    if fc is not None:
        fc = fc[fc["dataset_id"] == ds].drop(columns=["dataset_id", "power_now"], errors="ignore")
        sc = sc.merge(fc.drop_duplicates("time_stamp"), on="time_stamp", how="left")
    for c in ("wind", "power", "pred_ml", "pred_phys", *RAW_COLS.values()):
        if c not in sc:
            sc[c] = np.nan
    if "in_event_window" not in sc:
        sc["in_event_window"] = False
    return sc.sort_values("time_stamp").reset_index(drop=True)


# ================================================================ future: health trend
@st.cache_data
def trend_table(ds):
    """Slope (score/h) and hours to threshold for every subsystem and time step."""
    data = load_twin_data(ds)
    subs = [c for c in SUBSYSTEMS if c in data]
    if fd is None or not subs:
        return None, None
    thr = float(data["threshold"].iloc[0])
    return fd.health_trend(data[subs], thr)


def get_trend(ds, data):
    """Health trend: cached for recorded datasets, computed live for the synthetic turbine."""
    if ds != SYN:
        return trend_table(ds)
    subs = [c for c in SUBSYSTEMS if c in data]
    if fd is None or not subs or data.empty:
        return None, None
    return fd.health_trend(data[subs].reset_index(drop=True), float(data["threshold"].iloc[0]))


# ================================================================ maintenance advisor (rule-based)
# What to do when the twin flags a component: likely causes, actions NOW, and how to PREVENT it.
# Guidance from typical wind-turbine maintenance practice - always confirm with the OEM manual.
MAINT = {
    "Gearbox": dict(
        causes=["Bearing wear or pitting on the high-speed shaft", "Oil degraded, contaminated or low",
                "Oil cooler / filter blocked", "Misalignment between gearbox and generator"],
        now=["Check gearbox oil level, temperature and filter differential pressure",
             "Take an oil sample: particle count, water content, iron / wear metals",
             "Inspect the high-speed shaft bearing (temperature, noise, portable vibration check)",
             "Check oil cooler fans and the cooling circuit"],
        critical=["Derate the turbine to reduce the gearbox load until inspected",
                  "Stop the turbine if the bearing temperature keeps rising"],
        prevent=["Oil analysis every 6 months; change oil and filters on schedule",
                 "Endoscope inspection of gears and bearings once a year",
                 "Keep oil cooler and filters clean; check shaft alignment",
                 "Add vibration (CMS) sensors on the gearbox for earlier warning"]),
    "Generator bearings": dict(
        causes=["Lubrication failure (too little, too much or old grease)", "Bearing wear or brinelling",
                "Shaft currents causing electrical erosion", "Misalignment with the gearbox"],
        now=["Compare drive-end and non-drive-end bearing temperatures",
             "Check the grease system: grease level, dosing, leaks",
             "Listen / measure vibration at both bearings",
             "Check the shaft earthing brush and insulation"],
        critical=["Derate the turbine to lower the bearing load",
                  "Stop the turbine before the bearing seizes if the temperature keeps rising"],
        prevent=["Follow the regreasing interval; use the specified grease",
                 "Inspect earthing brushes and bearing insulation every service",
                 "Laser alignment of generator and gearbox after major work",
                 "Vibration monitoring of the generator bearings"]),
    "Generator windings": dict(
        causes=["Generator cooling reduced (fan, filter, air duct)", "Phase imbalance or overload",
                "Insulation ageing"],
        now=["Check stator temperature of all three phases (is one phase hotter?)",
             "Inspect generator cooling: fans, filters, air ducts",
             "Check phase currents for imbalance"],
        critical=["Derate the turbine to reduce copper losses (heat ~ current²)",
                  "Stop and test insulation if one phase is far hotter than the others"],
        prevent=["Clean generator filters and ducts every service",
                 "Insulation resistance test once a year",
                 "Keep phase currents balanced; check cable terminations"]),
    "Transformer": dict(
        causes=["Cooling / ventilation of the transformer cell reduced", "Overload or harmonics",
                "Loose connection or insulation problem"],
        now=["Compare the three transformer phase temperatures",
             "Inspect transformer ventilation, fans and the cell air flow",
             "Check load current and grid voltage; look for loose connections (thermal camera)"],
        critical=["Derate the turbine to lower the transformer load",
                  "Disconnect and inspect if one phase keeps heating up (fire risk)"],
        prevent=["Thermographic inspection of connections once a year",
                 "Clean the transformer cell and check the ventilation",
                 "Oil / insulation test as specified by the manufacturer"]),
    "Hydraulic group": dict(
        causes=["Oil leak or low oil level", "Pump running too often (internal leakage)",
                "Accumulator pre-charge pressure lost", "Oil cooler or filter blocked"],
        now=["Check hydraulic oil level and look for leaks (hoses, seals, cylinders)",
             "Check the pump running time / start count and system pressure",
             "Check accumulator pre-charge pressure",
             "Inspect oil filter and cooler"],
        critical=["Check that the pitch system can still feather the blades (safety)",
                  "Stop the turbine if pressure cannot be held"],
        prevent=["Replace hydraulic filters and test oil on schedule",
                 "Check accumulator pre-charge every service",
                 "Replace ageing hoses and seals before they leak"]),
    "Hub / pitch": dict(
        causes=["Pitch actuator or hydraulic pitch problem", "Hub controller overheating",
                "Pitch angle sensor / slip ring fault"],
        now=["Compare pitch angle response of each blade with the expected angle",
             "Inspect pitch hydraulics / drives: pressure, valves, accumulators",
             "Check hub controller temperature and the slip-ring connections"],
        critical=["A pitch fault is a SAFETY issue: be ready to stop the turbine",
                  "Do not run the turbine at high wind until the pitch is verified"],
        prevent=["Pitch system function test every service (all blades, emergency feathering)",
                 "Clean and inspect slip rings and hub cabinet cooling",
                 "Calibrate pitch angle sensors"]),
    "Converter": dict(
        causes=["Converter cooling (water pump, fans, coolant) reduced", "IGBT module ageing",
                "High switching / load"],
        now=["Check converter coolant temperature, flow and level",
             "Inspect cooling pump, fans and filters",
             "Read the converter fault log; compare the IGBT phase temperatures"],
        critical=["Derate the turbine to reduce converter current",
                  "Stop if IGBT temperatures approach their trip limit"],
        prevent=["Coolant check and filter cleaning every service",
                 "Thermal inspection of converter modules and busbars",
                 "Keep cabinet seals and dust filters in good condition"]),
    "Nacelle cabinets": dict(
        causes=["Cabinet ventilation / cooling reduced", "Ambient heat in the nacelle",
                "Electronic component ageing"],
        now=["Check cabinet fans, filters and the cooling-water circuit",
             "Read the controller fault log",
             "Check nacelle ventilation and the nacelle air temperature"],
        critical=["Reduce load if control electronics approach their temperature limit"],
        prevent=["Clean cabinet filters every service",
                 "Check fans and heaters (also for condensation in winter)",
                 "Thermographic check of cabinets once a year"]),
    "Electrical balance": dict(
        causes=["Loose or corroded cable connection", "Grid voltage imbalance",
                "Generator or converter phase problem"],
        now=["Check phase currents and voltages",
             "Thermal camera check of cable terminations and busbars",
             "Check grid-side measurements with the grid operator"],
        critical=["Stop the turbine if the imbalance is large (protection may trip anyway)"],
        prevent=["Torque check of electrical connections every service",
                 "Annual thermography of the power path"]),
    "Power performance": dict(
        causes=["Anemometer or wind-vane error (wrong yaw / wrong wind reading)",
                "Pitch misalignment or blade damage / dirt / ice", "Power limitation active"],
        now=["Compare power with neighbouring turbines in the same wind",
             "Check anemometer and wind-vane calibration and yaw alignment",
             "Check for derating / curtailment settings; inspect blades (drone or binoculars)"],
        critical=["If a blade is damaged: stop the turbine and inspect"],
        prevent=["Calibrate wind sensors; check yaw misalignment yearly",
                 "Blade inspection and cleaning / leading-edge repair",
                 "Power-curve monitoring against the twin (this dashboard)"]),
    "Whole turbine (AE)": dict(
        causes=["Several signals behave unusually together - often an early, not yet specific fault",
                "Sensor fault or data problem"],
        now=["Look at the sensors with the largest deviation (listed below) and inspect that component",
             "Check whether a sensor is frozen, jumping or out of range"],
        critical=["Treat as the component of the most deviating sensor"],
        prevent=["Keep sensors calibrated; replace faulty sensors quickly"]),
}
URGENCY = {"Act now": (RED, "#FDE8E8", 0), "Plan inspection": (AMBER, "#FEF3C7", 1), "Monitor": (BLUE, BLUE_LIGHT, 2)}


def sensor_name(col):
    names = getattr(fd, "SENSOR_NAMES", {}) if fd else {}
    return names.get(col, col.replace("_avg", "").replace("_", " "))


def advisory(ds, data, cursor, sub_cols):
    """Components that need attention at twin time `cursor`, with urgency and evidence."""
    cur = data.iloc[cursor]
    thr = float(cur.get("threshold", np.nan))
    if pd.isna(thr):
        return []
    recent = data.iloc[max(0, cursor - 35):cursor + 1]                  # last 6 h
    valid = {s: cur.get(s) for s in sub_cols if not pd.isna(cur.get(s))}
    top = max(valid, key=valid.get) if valid else None
    _, hrs = get_trend(ds, data)
    subsys = getattr(fd, "SUBSYSTEMS", {}) if fd else {}
    items = []
    for s, sc in valid.items():
        t_alarm = hrs[s].iloc[cursor] if hrs is not None and s in hrs else np.nan
        sustained = int((recent[s] >= 0.6 * thr).sum())
        alarm_on = bool(cur.get("alarm", False))
        if alarm_on and (s == top or sc >= thr):           # only a confirmed (sustained) alarm means "act now"
            level = "Act now"
        elif sc >= thr or sustained >= STEPS_PER_HOUR or (not pd.isna(t_alarm) and t_alarm <= 24 and sc > 0.5 * thr):
            level = "Plan inspection"
        elif sc >= 0.4 * thr:
            level = "Monitor"
        else:
            continue
        # how long has it been elevated (consecutive steps >= 60 % of threshold)?
        above = (data[s].iloc[:cursor + 1] >= 0.6 * thr).to_numpy()[::-1]
        run = int(np.argmin(above)) if not above.all() else len(above)
        ev = []
        for sens in subsys.get(s, []):
            zc = f"z_{sens}"
            if zc in data:
                zm = float(data[zc].iloc[max(0, cursor - 35):cursor + 1].mean())
                if not pd.isna(zm) and zm > 1:
                    ev.append((zm, sens))
        ev = sorted(ev, reverse=True)[:2]
        evidence = [f"{sensor_name(c)}: {z:.1f} σ {'below' if c == 'power_30_avg' else 'away from'} the expected value (last 6 h)"
                    if c in ("power_30_avg", "pitch_dev") else
                    f"{sensor_name(c)}: {z:.1f} σ hotter than expected (last 6 h)" for z, c in ev]
        evidence.append(f"Score {sc:.1f} of alarm level {thr:.1f} ({100 * sc / thr:.0f} %)"
                        + (f", elevated for {run / STEPS_PER_HOUR:.1f} h" if run else ""))
        if not pd.isna(t_alarm) and level != "Act now":
            evidence.append(f"Trend: reaches alarm level in about {t_alarm:.0f} h if it continues")
        items.append(dict(component=s, level=level, score=sc, evidence=evidence, **MAINT.get(s, MAINT["Whole turbine (AE)"])))
    return sorted(items, key=lambda d: (URGENCY[d["level"]][2], -d["score"]))


def advice_html(item):
    col, bg, _ = URGENCY[item["level"]]
    lst = lambda xs: "".join(f"<li style='margin:3px 0'>{x}</li>" for x in xs)
    now = (item["critical"] + item["now"]) if item["level"] == "Act now" else item["now"]
    when = {"Act now": "Do now (today)", "Plan inspection": "Plan within the next days",
            "Monitor": "Keep watching - check at the next service"}[item["level"]]
    return (f'<div class="card" style="border-left:5px solid {col};margin-bottom:12px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
            f'<div style="font-weight:800;font-size:16px">🛠 {item["component"]}</div>'
            f'<span class="pill" style="background:{bg};color:{col}">{item["level"]}</span></div>'
            f'<div class="muted" style="margin-bottom:8px">{" · ".join(item["evidence"])}</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;font-size:13px">'
            f'<div><b>Likely causes</b><ul style="margin:4px 0 0 16px;padding:0">{lst(item["causes"])}</ul></div>'
            f'<div><b style="color:{col}">{when}</b><ul style="margin:4px 0 0 16px;padding:0">{lst(now)}</ul></div>'
            f'<div><b style="color:{GREEN}">Prevent it</b><ul style="margin:4px 0 0 16px;padding:0">{lst(item["prevent"])}</ul></div>'
            f'</div></div>')


def advice_text(items, ds_text, when):
    lines = [f"Maintenance advice - {ds_text}", f"Twin time: {when}", ""]
    if not items:
        lines.append("All components within their normal range. Continue routine maintenance.")
    for it in items:
        now = (it["critical"] + it["now"]) if it["level"] == "Act now" else it["now"]
        lines += [f"[{it['level'].upper()}] {it['component']}", *[f"  - {e}" for e in it["evidence"]],
                  "  Likely causes:", *[f"    * {x}" for x in it["causes"]],
                  "  Actions now:", *[f"    [ ] {x}" for x in now],
                  "  Prevention:", *[f"    * {x}" for x in it["prevent"]], ""]
    lines.append("Rule-based guidance from typical maintenance practice - confirm with the OEM manual.")
    return "\n".join(lines)


def page_advisor(info, ds, data, cursor, sub_cols):
    items = advisory(ds, data, cursor, sub_cols)
    when = f"{pd.Timestamp(data.iloc[cursor]['time_stamp']):%d %b %Y %H:%M}"
    section_title("Recommended actions", "What to do now and how to prevent it - for every component the twin flags")
    if not items:
        show_html(f'<div class="card" style="border-left:5px solid {GREEN}"><b>✔ All components within their normal range.</b>'
                  f'<div class="muted" style="margin-top:4px">No action needed now. Keep routine maintenance: oil analysis, '
                  f'filter cleaning, greasing, pitch function test, sensor calibration.</div></div>')
    for it in items[:4]:
        show_html(advice_html(it))
    if len(items) > 4:
        st.caption(f"+ {len(items) - 4} more component(s) to monitor - see the download.")
    st.download_button("⬇ Download maintenance advice (.txt)", advice_text(items, info[ds]["text"], when),
                       file_name=f"maintenance_advice_dataset{ds}.txt")
    st.caption("Rule-based guidance from typical wind-turbine maintenance practice, triggered by the twin's "
               "detection. Confirm with the manufacturer's manual and the maintenance team.")


# ================================================================ plotly styling
def style(fig, title=None, height=320, ytitle=None, xtitle=None):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK, family=FONT), x=0.01) if title else None,
        height=height, margin=dict(l=10, r=10, t=46 if title else 10, b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(family=FONT, color=INK, size=12),
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        yaxis_title=ytitle, xaxis_title=xtitle, uirevision="twin")
    fig.update_xaxes(showgrid=True, gridcolor=LINE, zeroline=False, linecolor=LINE)
    fig.update_yaxes(showgrid=True, gridcolor=LINE, zeroline=False, linecolor=LINE)
    return fig


def power_chart(view, rated, sigma, now_ts, cur=None):
    k = 100.0 / rated
    fig = go.Figure()
    if sigma:
        band = 1.28 * sigma * k
        up, lo = (view["pred_ml"] * k + band), (view["pred_ml"] * k - band).clip(lower=0)
        fig.add_trace(go.Scatter(x=view["time_stamp"], y=up, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=view["time_stamp"], y=lo, line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(47,128,237,0.13)", name="Expected range (80 %)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=view["time_stamp"], y=view["power"] * k, name="Actual", line=dict(color=TEAL, width=2)))
    fig.add_trace(go.Scatter(x=view["time_stamp"], y=view["pred_ml"] * k, name="Twin (expected)",
                             line=dict(color=BLUE, width=2, dash="dash")))
    # ---- forecast fan for the next 3 hours (made at "now")
    if cur is not None and not pd.isna(cur.get("f6", np.nan)):
        hs = [h for h in HORIZONS if not pd.isna(cur.get(f"f{h}", np.nan))]
        xs = [pd.Timestamp(now_ts)] + [pd.Timestamp(now_ts) + timedelta(minutes=10 * h) for h in hs]
        p0 = cur.get("power", np.nan) * k
        fy = [p0] + [cur[f"f{h}"] * k for h in hs]
        lo = [p0] + [cur[f"lo{h}"] * k for h in hs]
        hi = [p0] + [cur[f"hi{h}"] * k for h in hs]
        fig.add_trace(go.Scatter(x=xs, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=lo, line=dict(width=0), fill="tonexty", fillcolor="rgba(139,92,246,0.15)",
                                 name="Forecast range (80 %)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=fy, name="Forecast (next 3 h)", mode="lines+markers",
                                 line=dict(color=PURPLE, width=2, dash="dot"), marker=dict(size=5)))
    fig.add_vline(x=now_ts, line=dict(color="#0B3B5C", width=1.5, dash="dot"),
                  annotation_text="Now", annotation_position="top", annotation_bgcolor="#0B3B5C",
                  annotation_font=dict(color="white", size=11))
    has_fc = cur is not None and not pd.isna(cur.get("f6", np.nan))
    return style(fig, "Power: actual vs digital twin" + (", with 3-hour forecast" if has_fc else ""), 355,
                 "% of rated power")


def anomaly_chart(view, sub_cols, thr, show_window, per_subsystem=False):
    fig = go.Figure()
    if show_window and view["in_event_window"].any():
        w = view.loc[view["in_event_window"], "time_stamp"]
        fig.add_vrect(x0=w.min(), x1=w.max(), fillcolor=RED, opacity=0.07, line_width=0,
                      annotation_text="injected fault" if "fault_label" in view else "recorded fault window", annotation_position="top right",
                      annotation_font=dict(size=10, color=RED))
    if per_subsystem:
        top = view[sub_cols].max().sort_values(ascending=False).index[:5] if len(view) else []
        for col, colr in zip(top, [BLUE, RED, TEAL, PURPLE, AMBER]):
            fig.add_trace(go.Scatter(x=view["time_stamp"], y=view[col], name=col, line=dict(width=1.6, color=colr)))
    else:
        fig.add_trace(go.Scatter(x=view["time_stamp"], y=view["anomaly_score"], name="Anomaly score",
                                 line=dict(color=TEAL, width=2), fill="tozeroy", fillcolor="rgba(14,159,142,0.08)"))
    fig.add_hline(y=thr, line=dict(color=INK, dash="dash", width=1), annotation_text=f"threshold {thr:.1f}",
                  annotation_position="top left", annotation_font=dict(size=10, color=MUTED))
    al = view[view["alarm"].fillna(False).astype(bool)]
    if len(al):
        fig.add_trace(go.Scatter(x=al["time_stamp"], y=al["anomaly_score"], mode="markers", name="Anomaly",
                                 marker=dict(color=RED, size=7, line=dict(color="white", width=1))))
    title = "Anomaly scores by subsystem (top 5)" if per_subsystem else "Anomaly detection"
    return style(fig, title, 340, "smoothed z-score")


def power_curve_chart(full, cur, curve, rated):
    k = 100.0 / rated
    fig = go.Figure()
    hist = full.dropna(subset=["wind", "power"])
    fig.add_trace(go.Scattergl(x=hist["wind"], y=hist["power"] * k, mode="markers", name="History",
                               marker=dict(size=3, color=GREY, opacity=0.35)))
    if curve is not None:
        v = np.linspace(0, max(float(np.nanmax(curve.v_)), 1), 200)
        fig.add_trace(go.Scatter(x=v, y=curve.predict(v) * k, name="Reference power curve", line=dict(color=RED, width=2)))
    if not pd.isna(cur.get("wind")) and not pd.isna(cur.get("power")):
        fig.add_trace(go.Scatter(x=[cur["wind"]], y=[cur["power"] * k], mode="markers", name="Now",
                                 marker=dict(size=15, color=BLUE, line=dict(color="white", width=2))))
    fig = style(fig, "Operating point on the power curve", 340, "% of rated power", "Wind speed (m/s)")
    fig.update_layout(hovermode="closest")
    return fig


def scatter_chart(pw, rated):
    k = 100.0 / rated
    s = pw.dropna(subset=["power", "pred_ml"])
    s = s.sample(min(4000, len(s)), random_state=0) if len(s) else s
    fig = go.Figure(go.Scattergl(x=s["power"] * k, y=s["pred_ml"] * k, mode="markers",
                                 marker=dict(size=4, color=BLUE, opacity=0.35), name="Test points"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=INK, dash="dash", width=1), name="Perfect"))
    fig = style(fig, "Actual vs predicted power (unseen data)", 340, "Predicted (% of rated)", "Actual (% of rated)")
    fig.update_layout(hovermode="closest")
    return fig


def health_bars(health):
    items = sorted([(k, v) for k, v in health.items() if not pd.isna(v)], key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=[colour_for(v) for _, v in items],
                           text=[f"{v:.0f}%" for _, v in items], textposition="auto"))
    fig.update_xaxes(range=[0, 100])
    fig = style(fig, "Component health now", 380, None, "Health (%)")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ================================================================ page sections
def current_trend(ds, data, cursor):
    """(subsystem, hours) of the subsystem expected to reach the alarm level soonest (<= 24 h)."""
    slope, hrs = get_trend(ds, data)
    if hrs is None:
        return None
    thr = float(data["threshold"].iloc[cursor])
    row, sc = hrs.iloc[cursor], data.iloc[cursor]
    cand = {c: v for c, v in row.items() if not pd.isna(v) and v <= 24 and sc.get(c, 0) > 0.5 * thr}
    if not cand:
        return None
    name = min(cand, key=cand.get)
    return name, cand[name]


def section_title(text, sub=None):
    st.markdown(f"<div style='font-size:20px;font-weight:800;color:{INK};margin:10px 0 2px'>{text}</div>"
                + (f"<div style='color:{MUTED};font-size:13px;margin-bottom:8px'>{sub}</div>" if sub else ""),
                unsafe_allow_html=True)


def current_state(data, cursor, sub_cols, rated):
    cur = data.iloc[cursor].to_dict()
    prev = data.iloc[max(0, cursor - STEPS_PER_HOUR)].to_dict()
    thr = float(cur.get("threshold", np.nan))
    health = {s: health_from_score(cur.get(s), thr) for s in sub_cols}
    valid = {k: v for k, v in health.items() if not pd.isna(v)}
    weakest = min(valid, key=valid.get) if valid else "-"
    overall = valid[weakest] if valid else np.nan
    return cur, prev, thr, health, overall, weakest


def make_insights(cur, prev, health, overall, weakest, rated, sigma, crit_limit, trend=None):
    k = 100.0 / rated
    out = []
    if not pd.isna(cur.get("f6", np.nan)):
        out.append(("info", f"Forecast next hour: <b>{cur['f6'] * k:.0f} %</b> of rated "
                            f"(80 % range {cur['lo6'] * k:.0f}-{cur['hi6'] * k:.0f} %)."))
    if trend:
        name, hrs = trend
        out.append(("warn", f"<b>{name}</b> trending up - alarm level in about <b>{hrs:.0f} h</b> if the trend continues."))
    p, e = cur.get("power"), cur.get("pred_ml")
    if not pd.isna(p) and not pd.isna(e):
        band = 1.28 * sigma * k if sigma else 3.0
        gap = (p - e) * k
        if abs(gap) <= band:
            out.append(("ok", f"Output matches the twin's expectation (gap {gap:+.1f} % of rated, band ±{band:.1f} %)."))
        elif gap < 0:
            out.append(("warn", f"Output is <b>{-gap:.1f} % of rated below</b> what the twin expects for this weather."))
        else:
            out.append(("info", f"Output is {gap:.1f} % of rated above expectation."))
    w0, w1 = prev.get("wind"), cur.get("wind")
    if not pd.isna(w0) and not pd.isna(w1):
        d = w1 - w0
        out.append(("info", f"Wind {'rising' if d > 0.5 else 'falling' if d < -0.5 else 'steady'} "
                            f"({w1:.1f} m/s, {d:+.1f} m/s in the last hour)."))
    if not pd.isna(overall):
        kind = "ok" if overall >= 70 else "warn" if overall >= 30 else "bad"
        out.append((kind, f"Weakest component: <b>{weakest}</b> at {overall:.0f} % health."))
    if bool(cur.get("alarm", False)):
        out.append(("bad", f"<b>Alarm active</b> - sustained deviation, inspect {weakest}."))
    elif cur.get("criticality", 0) > 0:
        out.append(("warn", f"Deviation building up (criticality {int(cur['criticality'])} of {crit_limit})."))
    else:
        out.append(("ok", "No active alarms."))
    return out


def sidebar_controls(info):
    """Sidebar widgets. The replay position is kept in st.session_state[pos_key] (not tied to a widget),
    so the page body can advance it frame by frame without re-running the whole app."""
    ds = st.sidebar.selectbox("Turbine dataset", list(info), format_func=lambda d: info[d]["text"])
    data = load_twin_data(ds)
    n = len(data)
    pos_key, seek_key = f"pos_{ds}", f"seek_{ds}"
    if pos_key not in st.session_state:
        st.session_state[pos_key] = min(n - 1, 6 * STEPS_PER_HOUR * 24)
    st.sidebar.markdown("**Replay**")
    playing = st.sidebar.toggle("Play", key="playing")
    speed = st.sidebar.select_slider("Speed (10-min steps per frame)", [1, 3, 6, 12, 36], value=6)

    def _seek():
        st.session_state[pos_key] = st.session_state[seek_key]
    st.session_state[seek_key] = int(st.session_state[pos_key])      # slider shows the position when paused
    st.sidebar.slider("Position", 0, n - 1, key=seek_key, on_change=_seek,
                      help="Drag to jump. While playing, the position shown in the page header is the live one.")
    window = st.sidebar.selectbox("Chart window", ["1 day", "3 days", "7 days", "Everything so far"], index=1)
    show_window = st.sidebar.checkbox("Show recorded fault window", value=True)
    steps = {"1 day": 144, "3 days": 432, "7 days": 1008}.get(window, n)
    return ds, data, pos_key, speed, steps, show_window, playing


@st.cache_resource(show_spinner=False)
def syn_models(base):
    return syn.train_models(base, find_farm_dir())


def syn_controls(info, model, crit_limit):
    """Sidebar for the live synthetic turbine. Returns the turbine object and display settings."""
    normals = [d for d in info if info[d]["label"] == "normal"] or list(info)
    base = st.sidebar.selectbox("Imitate this healthy turbine", normals,
                                format_func=lambda d: f"Dataset {d} · Asset {info[d]['asset']}")
    scen = st.sidebar.selectbox("Weather", list(syn.SCENARIOS))
    tb = st.session_state.get("syn_tb")
    if tb is None or st.session_state.get("syn_base") != base:
        with st.spinner(f"Training the synthetic turbine on dataset {base} (first time about 30-60 s) ..."):
            m = syn_models(base)
        thr = float(load_twin_data(base)["threshold"].iloc[0])
        tb = syn.SyntheticTurbine(m, scen, thr, crit_limit, power_model=model,
                                  seed=int(time.time()) % 100000, warmup_days=3)
        st.session_state["syn_tb"], st.session_state["syn_base"] = tb, base
    tb.scenario = scen
    st.sidebar.markdown("**Live simulation**")
    playing = st.sidebar.toggle("Play", key="playing")
    speed = st.sidebar.select_slider("Speed (10-min steps per frame)", [1, 3, 6, 12, 36], value=6)
    c1, c2 = st.sidebar.columns(2)
    if c1.button("⏭ +1 day"):
        tb.step(144)
    if c2.button("↺ Restart"):
        st.session_state.pop("syn_tb", None)
        st.rerun()
    with st.sidebar.expander("💉 Inject a fault", expanded=True):
        name = st.selectbox("Fault", list(syn.FAULTS))
        sev = st.selectbox("Severity", list(syn.SEVERITY), index=1,
                           help="Mild stays below the alarm level (trend warning), Moderate alarms after a while, Severe fast")
        dev = st.selectbox("How it develops", list(syn.DEVELOPMENT), index=1)
        if st.button("Inject fault"):
            tb.inject(name, sev, dev)
        for f in tb.faults:
            st.caption(f"🔴 {f['name']} · {f['severity']} · {f['development'].lower()} · since "
                       f"{pd.Timestamp(f['start']):%d %b %H:%M} ({f['equivalent']})")
        if tb.faults and st.button("🔧 Repair (clear faults)"):
            tb.clear_faults()
    window = st.sidebar.selectbox("Chart window", ["1 day", "3 days", "7 days", "Everything so far"], index=1)
    show_window = st.sidebar.checkbox("Show injected fault period", value=True)
    steps = {"1 day": 144, "3 days": 432, "7 days": 1008}.get(window, 10 ** 6)
    st.sidebar.download_button("⬇ Download synthetic data (CSV)", tb.frame().to_csv(index=False),
                               file_name="synthetic_turbine.csv")
    return tb, base, steps, show_window, playing, speed


def page_overview(info, ds, data, cursor, steps, show_window, model, sigma, crit_limit):
    sub_cols = [c for c in SUBSYSTEMS if c in data]
    rated = model["rated_power"] if model else max(float(data["power"].quantile(0.999)), 1e-9)
    k = 100.0 / rated
    cur, prev, thr, health, overall, weakest = current_state(data, cursor, sub_cols, rated)
    status, scol = turbine_status(cur)
    view = data.iloc[max(0, cursor - steps + 1):cursor + 1]
    so_far = data.iloc[:cursor + 1]
    spark = data.iloc[max(0, cursor - 35):cursor + 1]

    live = ds == SYN
    show_html(hero_html(info[ds]["text"], f"{pd.Timestamp(cur['time_stamp']):%d %b %Y %H:%M}", cursor + 1,
                        len(data), mode="Live · SYNTHETIC data" if live else "Replay mode",
                        dot=PURPLE if live else GREEN), height=186)

    cards = [
        dict(icon="⚡", icon_bg=TEAL_LIGHT, title="Power output", value=f"{fmt(cur.get('power', np.nan) * k, 0)} %",
             delta=delta_html(cur.get("power", np.nan) * k, prev.get("power", np.nan) * k, " pts", absolute=True),
             spark=sparkline_svg(spark["power"] * k, GREEN)),
        dict(icon="🌬", icon_bg=BLUE_LIGHT, title="Wind speed", value=f"{fmt(cur.get('wind'), 1)} m/s",
             delta=delta_html(cur.get("wind"), prev.get("wind")), spark=sparkline_svg(spark["wind"], BLUE)),
        dict(icon="🌡", icon_bg="#FDECDC", title="Ambient temperature", value=f"{fmt(cur.get('ambient_temp'), 1)} °C",
             delta=delta_html(cur.get("ambient_temp"), prev.get("ambient_temp"), " °C", absolute=True),
             spark=sparkline_svg(spark["ambient_temp"], PURPLE)),
        dict(icon="✔" if status == "Normal" else "⚠", icon_bg=TEAL_LIGHT if status == "Normal" else "#FDE8E8",
             title="Turbine status", value=status, big_colour=scol,
             delta=('<span class="muted">No active faults</span>' if status == "Normal"
                    else f'<span class="muted">{weakest} · criticality {int(cur.get("criticality", 0))}</span>')),
    ]
    show_html(kpi_cards_html(cards), height=178)

    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.plotly_chart(power_chart(view, rated, sigma, cur["time_stamp"], cur), use_container_width=True)
    with c2:
        ins = make_insights(cur, prev, health, overall, weakest, rated, sigma, crit_limit,
                            current_trend(ds, data, cursor))
        adv = advisory(ds, data, cursor, sub_cols)
        if adv:
            ins.insert(0, ("bad" if adv[0]["level"] == "Act now" else "warn",
                           f"<b>{adv[0]['level']}:</b> {adv[0]['component']} - {adv[0]['now'][0].lower()}. "
                           f"See <b>Anomaly Detection → Recommended actions</b>."))
        show_html(insights_html(ins, min_h=345), height=365)

    events = event_list(so_far, sub_cols, sigma)
    c3, c4 = st.columns([1.35, 1])
    with c3:
        st.plotly_chart(anomaly_chart(view, sub_cols, thr, show_window), use_container_width=True)
    with c4:
        show_html(anomaly_table_html(events), height=350)

    c5, c6 = st.columns([1, 1])
    with c5:
        readings = [("🌀", "Rotor speed", f"{fmt(cur.get('rotor_rpm'), 1)} rpm"),
                    ("⚙", "Generator speed", f"{fmt(cur.get('gen_rpm'), 0)} rpm"),
                    ("⚡", "Power output", f"{fmt(cur.get('power', np.nan) * k, 0)} % of rated"),
                    ("🌡", "Nacelle temperature", f"{fmt(cur.get('nacelle_temp'), 1)} °C"),
                    ("↻", "Pitch angle", f"{fmt(cur.get('pitch'), 1)} °")]
        show_html(health_card_html(overall, weakest, readings), height=300)
    with c6:
        show_html(alerts_html(events), height=300)


def page_power(model, data, cursor, sigma):
    section_title("Power Prediction", "How well the twin predicts power from the weather, on data it never saw")
    pm = load_json(os.path.join(POWER_DIR, "metrics.json"))
    pw = load_csv(os.path.join(POWER_DIR, "test_predictions.csv"))
    if pm:
        ml, ph = pm["ml_gradient_boosting"], pm["physics_power_curve"]
        gain = 100 * (1 - ml["RMSE"] / ph["RMSE"])
        show_html(metric_cards_html([
            ("MAE (ML twin)", f"{ml['nMAE_%']:.2f} %", f"of rated · physics {ph['nMAE_%']:.2f} %"),
            ("RMSE (ML twin)", f"{ml['nRMSE_%']:.2f} %", f"of rated · physics {ph['nRMSE_%']:.2f} %"),
            ("R²", f"{ml['R2']:.3f}", f"physics {ph['R2']:.3f}"),
            ("Improvement", f"{gain:.0f} %", "lower RMSE than physics curve")]), height=118)
    rated = model["rated_power"] if model else 1.0
    c1, c2 = st.columns(2)
    with c1:
        if pw is not None:
            normal = pw[pw.get("event_label", pd.Series("normal", index=pw.index)).astype(str).str.lower() == "normal"]
            st.plotly_chart(scatter_chart(normal if len(normal) else pw, rated), use_container_width=True)
    with c2:
        cur = data.iloc[cursor].to_dict()
        st.plotly_chart(power_curve_chart(data.iloc[:cursor + 1], cur, model["power_curve"] if model else None, rated),
                        use_container_width=True)
    fm = load_json(os.path.join(FORECAST_DIR, "metrics.json"))
    section_title("Short-term forecast", "Power 10 min to 3 h ahead from recent history, compared with persistence "
                                         "(\"the future equals now\")")
    if fm:
        hz = fm["horizons"]
        show_html(metric_cards_html([
            (f"{lab} ahead", f"{v['nRMSE_%']:.1f} %",
             f"nRMSE · skill {v['skill_vs_persistence']:+.2f} · band hits {100 * v['interval_coverage_80']:.0f} %")
            for lab, v in hz.items()]), height=118)
        labs = list(hz)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=labs, y=[hz[l]["nRMSE_persistence_%"] for l in labs], name="Persistence",
                                 mode="lines+markers", line=dict(color=GREY, width=2)))
        fig.add_trace(go.Scatter(x=labs, y=[hz[l]["nRMSE_%"] for l in labs], name="Twin forecast",
                                 mode="lines+markers", line=dict(color=PURPLE, width=2.5)))
        fig = style(fig, "Forecast error vs horizon (unseen normal data)", 320, "nRMSE (% of rated)", "Horizon")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Skill > 0 means the twin beats persistence. Without weather-forecast data the advantage "
                   "shrinks as the horizon grows.")
    else:
        st.info("Run `python forecast_model.py` to add the short-term forecast.")
    section_title("What-if calculator", "Expected power for any weather you choose")
    page_whatif(model)


def page_whatif(model):
    if model is None:
        st.error("No power model found. Run `python power_model.py` first.")
        return
    feats, mdl, curve, rated = model["features"], model["model"], model["power_curve"], model["rated_power"]
    c1, c2, c3 = st.columns(3)
    wind = c1.slider("Wind speed (m/s)", 0.0, 25.0, 8.0, 0.5)
    temp = c2.slider("Ambient temperature (°C)", -10.0, 40.0, 15.0, 1.0)
    direction = c3.slider("Wind direction (°)", 0, 359, 270, 5)
    c4, c5, c6 = st.columns(3)
    turb = c4.slider("Gustiness (1-h std of wind speed, m/s)", 0.0, 3.0, 0.5, 0.1)
    hour = c5.slider("Hour of day", 0, 23, 14)
    month = c6.slider("Month", 1, 12, 6)
    doy = int(30.4 * (month - 1) + 15)
    X = weather_features(wind, temp, direction, turb, hour, doy, feats)
    p_ml = float(np.clip(mdl.predict(X)[0], 0, rated * 1.02))
    p_ph = float(curve.predict(np.array([X["wind_norm"].iloc[0]]))[0])
    rho = P_STD / (R_DRY_AIR * (temp + 273.15))

    def phys(w, t):
        """IEC power curve with air-density correction: wind is normalised to standard air density."""
        r = P_STD / (R_DRY_AIR * (t + 273.15))
        return 100 * curve.predict(np.atleast_1d(w) * (r / RHO_REF) ** (1 / 3)) / rated

    p_ref = float(phys(wind, 15.0)[0])
    t_effect = (100 * p_ph / rated - p_ref) if p_ref > 0.5 else 0.0
    show_html(metric_cards_html([
        ("Expected power · ML twin", f"{100 * p_ml / rated:.1f} %", "of rated power (learned from data)"),
        ("Expected power · power curve", f"{100 * p_ph / rated:.1f} %", "IEC physics baseline, air-density corrected"),
        ("Air density", f"{rho:.3f} kg/m³", f"{100 * (rho / RHO_REF - 1):+.1f} % vs standard"),
        ("Temperature effect", f"{t_effect:+.1f} pts", "power curve vs the same wind at 15 °C")]), height=118)
    v = np.arange(0, 25.01, 0.25)
    fig = go.Figure()
    temps = sorted({-5.0, float(temp), 35.0})
    colours = {-5.0: BLUE, 35.0: RED}
    for t in temps:
        mine = t == float(temp)
        fig.add_trace(go.Scatter(x=v, y=phys(v, t), name=f"Power curve {t:.0f} °C" + (" (your input)" if mine else ""),
                                 line=dict(color=TEAL if mine else colours.get(t, GREY), width=2.6 if mine else 1.3)))
    Xs = pd.concat([weather_features(w, temp, direction, turb, hour, doy, feats) for w in v], ignore_index=True)
    fig.add_trace(go.Scatter(x=v, y=100 * np.clip(mdl.predict(Xs), 0, rated * 1.02) / rated, name=f"ML twin {temp:.0f} °C",
                             line=dict(color=PURPLE, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[wind], y=[100 * p_ph / rated], mode="markers", name="Your input",
                             marker=dict(size=14, color=AMBER, line=dict(color="white", width=2))))
    fig = style(fig, "Expected power vs wind speed - effect of temperature (air density)", 380, "% of rated power",
                "Wind speed (m/s)")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cold air is denser, so the same wind gives more power (P = ½ ρ A Cp v³). The physics power curve applies "
               "this through air density; the ML twin learns mostly from wind speed and reacts only weakly to "
               "temperature, because temperature changes the power by a few percent at most and is mixed with "
               "seasonal effects in the data. Above rated wind both are flat (the pitch control limits the power).")


def page_anomaly(info, ds, data, cursor, steps, show_window, sigma):
    section_title("Anomaly Detection", "Normal-behaviour models for 11 subsystems + autoencoder; alarm after sustained deviation")
    sub_cols = [c for c in SUBSYSTEMS if c in data]
    view = data.iloc[max(0, cursor - steps + 1):cursor + 1]
    thr = float(data.iloc[cursor]["threshold"])
    st.plotly_chart(anomaly_chart(view, sub_cols, thr, show_window, per_subsystem=True), use_container_width=True)
    page_advisor(info, ds, data, cursor, sub_cols)
    section_title("Anomaly log")
    events = event_list(data.iloc[:cursor + 1], sub_cols, sigma)
    show_html(anomaly_table_html(events, n=8, min_h=0), height=110 + 46 * max(1, min(8, len(events))))
    cv = load_json(os.path.join(FAULT_DIR, "sweep_crossvalidation.json"))
    fs = load_json(os.path.join(FAULT_DIR, "summary.json"))
    if cv or fs:
        section_title("Detection performance (all 22 datasets)")
        items = []
        if cv:
            items += [("Faults detected (cross-validated)", f"{cv['detected']}/{cv['anomaly_datasets']}", "honest estimate"),
                      ("False alarms on healthy turbines", f"{cv['normal_with_false_alarm']}/{cv['normal_datasets']}", "normal datasets"),
                      ("Balanced accuracy", f"{cv.get('balanced_accuracy', float('nan')):.2f}", "0.5 = guessing")]
        if fs:
            items += [("Median warning", f"{fs.get('median_lead_time_days', '–')} d", "before recorded failure")]
        show_html(metric_cards_html(items), height=118)


def page_health(ds, data, cursor, model):
    section_title("Turbine Health", "Each component compared with its learned normal behaviour")
    sub_cols = [c for c in SUBSYSTEMS if c in data]
    rated = model["rated_power"] if model else 1.0
    cur, prev, thr, health, overall, weakest = current_state(data, cursor, sub_cols, rated)
    status, scol = turbine_status(cur)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        part_health = {}
        for part, subs in PARTS.items():
            hs = [(health[s], s) for s in subs if s in health and not pd.isna(health[s])]
            part_health[part] = min(hs) if hs else (np.nan, " / ".join(subs))
        pf = np.nan if pd.isna(cur.get("power")) else cur["power"] / rated
        show_html(BASE_CSS + '<div class="card">' + turbine_svg(part_health, pf, status, scol) + "</div>", height=520)
    with c2:
        st.plotly_chart(health_bars(health), use_container_width=True)
    slope, hrs = get_trend(ds, data)

    def trend_cols(sname):
        if slope is None or sname not in slope:
            return None, "–"
        sl, hr = slope[sname].iloc[cursor], hrs[sname].iloc[cursor]
        tta = "already above" if cur.get(sname, 0) >= thr else ("–" if pd.isna(hr) else
              f"~{hr:.0f} h" if hr <= 72 else "> 3 days")
        return (None if pd.isna(sl) else round(float(sl), 3)), tta

    rows = []
    for sname, h in sorted(health.items(), key=lambda kv: (pd.isna(kv[1]), kv[1])):
        sl, tta = trend_cols(sname)
        rows.append({"Subsystem": sname, "Health (%)": round(h, 0) if not pd.isna(h) else None,
                     "Score": round(float(cur.get(sname)), 2) if not pd.isna(cur.get(sname)) else None,
                     "Trend (score / h, last 24 h)": sl, "Time to alarm level": tta,
                     "State": "healthy" if h >= 70 else "watch" if h >= 30 else "critical" if not pd.isna(h) else "no data"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_reports():
    section_title("Reports", "Results of the power model and the fault detection for the project report")
    pm = load_json(os.path.join(POWER_DIR, "metrics.json"))
    if pm:
        rows = []
        for name, key in (("Physics power curve", "physics_power_curve"), ("Gradient Boosting (ML)", "ml_gradient_boosting")):
            m = pm.get(key, {})
            rows.append({"Model": name, "R²": round(m.get("R2", np.nan), 3), "nRMSE (% of rated)": round(m.get("nRMSE_%", np.nan), 2),
                         "nMAE (% of rated)": round(m.get("nMAE_%", np.nan), 2), "Test rows": m.get("n")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    res = load_csv(os.path.join(FAULT_DIR, "results_per_dataset.csv"), parse_time=False)
    if res is not None:
        keep = [c for c in ("dataset_id", "asset", "label", "fault", "detected", "lead_time_days",
                            "diagnosed_subsystem", "top_sensor", "diagnosis_correct", "false_alarm") if c in res]
        st.dataframe(res[keep], hide_index=True, use_container_width=True)
    imgs = [(os.path.join(POWER_DIR, "power_curve.png"), "Power curve"),
            (os.path.join(POWER_DIR, "feature_importance.png"), "Feature importance"),
            (os.path.join(FAULT_DIR, "sweep.png"), "Detection trade-off"),
            (os.path.join(FORECAST_DIR, "forecast_skill.png"), "Forecast skill"),
            (os.path.join(FORECAST_DIR, "forecast_example.png"), "1-hour-ahead forecast")]
    imgs = [x for x in imgs if os.path.isfile(x[0])]
    cols = st.columns(3)
    for i, (p, cap) in enumerate(imgs):
        cols[i % 3].image(p, caption=cap, use_container_width=True)
    ht = load_json(os.path.join(FAULT_DIR, "health_trend_summary.json"))
    if ht:
        st.markdown(f"**Health trend (time-to-alarm) evaluation:** trend warning on {ht['faults_with_trend_warning']}/"
                    f"{ht['anomaly_datasets']} faults, median {ht['median_hours_trend_before_alarm']} h before the alarm, "
                    f"{ht['normal_datasets_with_trend_warning']}/{ht['normal_datasets']} healthy turbines with a warning.")


GLOBAL_CSS = f"""<style>
.stApp {{background: linear-gradient(180deg,#EAF4FA 0%,#F6FAFD 60%,#F6FAFD 100%);}}
section[data-testid="stSidebar"] {{background:#ffffffee;border-right:1px solid {LINE};}}
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"], div[data-testid="stImage"] {{
  background:#fff;border-radius:16px;padding:8px 10px;border:1px solid {LINE};box-shadow:0 6px 20px rgba(30,80,120,.07);}}
div[data-testid="stSidebar"] div[role="radiogroup"] label {{padding:8px 10px;border-radius:10px;margin-bottom:2px}}
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{background:{TEAL_LIGHT};}}
#MainMenu, footer {{visibility:hidden;}}
[data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stDeployButton"] {{display:none !important;}}
{CARD_CSS}
.block-container {{padding-top:1.4rem;}}
</style>"""

BRAND = (f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0 14px'>"
         f"<svg width='34' height='40' viewBox='0 0 34 40'><g stroke='{TEAL}' stroke-width='3' stroke-linecap='round'>"
         f"<line x1='17' y1='14' x2='17' y2='39'/><line x1='17' y1='14' x2='17' y2='1'/><line x1='17' y1='14' x2='29' y2='21'/>"
         f"<line x1='17' y1='14' x2='5' y2='21'/></g><circle cx='17' cy='14' r='3' fill='{TEAL}'/></svg>"
         f"<div style='line-height:1.1'><div style='font-weight:800;font-size:18px;color:#0B3B5C'>Wind Turbine</div>"
         f"<div style='font-weight:700;font-size:17px;color:{TEAL}'>Digital Twin</div></div></div>")

PAGES = ["🏠  Overview", "⚡  Power Prediction", "⚠️  Anomaly Detection", "💚  Turbine Health", "🔗  Hybrid Twin",
         "🧊  3D Twin", "📄  Reports"]
LIVE_PAGES = PAGES[0], PAGES[2], PAGES[3], PAGES[4]          # pages that follow the replay position


# ================================================================ hybrid twin (learning model + physics model)
PHYS_DIR = os.path.join(HERE, "outputs_physics")
PART_NAMES = {"gearbox": "Gearbox", "generator": "Generator", "transformer": "Transformer",
              "hub": "Hub / pitch + hydraulics", "blades": "Blades (power performance)",
              "converter": "Converter / cabinets", "nacelle": "Whole turbine"}


@st.cache_resource(show_spinner=False)
def physics_params(asset):
    """Physics model tuned for one turbine (cached in outputs_physics/params_Axx.json)."""
    farm = find_farm_dir()
    if pt is None or farm is None:
        return None
    try:
        return pt.get_params(int(asset), farm, PHYS_DIR)
    except Exception as e:                       # shown on the page instead of crashing
        return {"error": str(e)}


@st.cache_data(show_spinner=False)
def physics_data(ds, asset):
    p = physics_params(asset)
    farm = find_farm_dir()
    if not p or "error" in p or farm is None:
        return None
    path = os.path.join(farm, "datasets", f"{ds}.csv")
    return pt.run(path, p) if os.path.isfile(path) else None


def ml_part_health(df, part):
    thr = pd.to_numeric(df["threshold"], errors="coerce").to_numpy(float)
    hs = [np.clip(100 * (1 - pd.to_numeric(df[s], errors="coerce").to_numpy(float) / thr), 0, 100)
          for s in PARTS[part] if s in df]
    if not hs:
        return np.full(len(df), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmin(np.vstack(hs), axis=0)


def with_physics(info, ds, data):
    """data + physics columns + learning/physics/combined health per part.
    Returns (frame, params, message); params is None when physics is not available."""
    df = data.copy()
    msg, p = None, None
    if pt is None:
        msg = "physics_twin.py not found - put it next to digital_twin_app.py."
    elif ds == SYN:
        msg = "The physics model needs the real sensor temperatures - switch the data source to Recorded data."
    else:
        try:
            asset = int(info.get(ds, {}).get("asset"))
        except (TypeError, ValueError):
            asset = None
        if asset is None:
            msg = "Unknown turbine for this dataset."
        else:
            with st.spinner("Tuning the physics model for this turbine (first time only, ~10-30 s) ..."):
                p = physics_params(asset)
                ph = None if (p is None or "error" in p) else physics_data(ds, asset)
            if p is None:
                msg = "Wind Farm A dataset not found - copy the CARE_To_Compare folder into this project folder."
            elif "error" in p:
                msg, p = f"The physics model could not be tuned: {p['error']}", None
            elif ph is None:
                msg, p = f"Raw file for dataset {ds} not found in the Wind Farm A folder.", None
            else:
                df = df.merge(ph, on="time_stamp", how="left")
    for part in PARTS:
        ml = ml_part_health(df, part)
        ph = pd.to_numeric(df[f"ph_{part}"], errors="coerce").to_numpy(float) if f"ph_{part}" in df else np.full(len(df), np.nan)
        df[f"ml_{part}"], df[f"phh_{part}"] = ml, ph
        df[f"cb_{part}"] = pt.combine(ml, ph) if pt is not None else ml
    return df, p, msg


def _fmt_h(h):
    return "–" if h is None or pd.isna(h) else f"{h:.0f} %"


def hybrid_cards_html(rows):
    pill = {"🟢": GREEN, "🟡": AMBER, "🔴": RED}
    cells = []
    for r in rows:
        col = pill.get(r["Combined"][:1], GREY)
        cells.append(
            f'<div class="card" style="flex:1;min-width:240px;border-top:5px solid {col}">'
            f'<div style="font-weight:800;font-size:17px">{r["Part"]}</div>'
            f'<div style="margin:6px 0 10px"><span class="pill" style="background:{col};color:white">{r["Combined"]}</span></div>'
            f'<div class="muted">Learning model (Python)</div><div style="font-weight:700;margin-bottom:6px">{r["Learning model"]}</div>'
            f'<div class="muted">Physics model (MATLAB equations)</div><div style="font-weight:700;margin-bottom:8px">{r["Physics model"]}</div>'
            f'<div style="font-size:13px;line-height:1.45">{r["Why"]}</div></div>')
    return f'<div style="display:flex;gap:14px;flex-wrap:wrap">{"".join(cells)}</div>'


def page_hybrid(info, ds, data, cursor, steps, show_window):
    section_title("Hybrid Twin", "Two opinions for each part: the learning model (Python) and the physics model "
                                 "(same equations as the MATLAB/Simulink twin)")
    df, p, msg = with_physics(info, ds, data)
    if p is None:
        st.info(msg)
        return
    cur = df.iloc[cursor].to_dict()
    rows = []
    for part in PARTS:
        hm, hp = cur.get(f"ml_{part}"), cur.get(f"phh_{part}")
        ws = pt.worst_sensor(cur, part, p) if part in pt.PHYS_PARTS else None
        if part not in pt.PHYS_PARTS:
            phys = "not modelled"
        elif bool(cur.get("low_load", False)):
            phys = "not judged (turbine nearly stopped)"
        elif ws is None:
            phys = "no data"
        else:
            phys = f"{_fmt_h(hp)} · {ws[1]:+.1f} °C vs expected (normal ±{ws[2]:.1f} °C)"
        rows.append({"Part": PART_NAMES.get(part, part), "Combined": pt.verdict(hm, hp),
                     "Learning model": _fmt_h(hm), "Physics model": phys,
                     "Why": pt.explain(part, hm, hp, ws, bool(cur.get("low_load", False))),
                     "_phys": part in pt.PHYS_PARTS, "_key": part})
    show_html(hybrid_cards_html([r for r in rows if r["_phys"]]))
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    others = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_") and k not in ("Physics model", "Why")}
                           for r in rows if not r["_phys"]])
    st.caption("Parts without a physics model - the learning model decides alone:")
    st.dataframe(others, use_container_width=True, hide_index=True)

    # ---- charts for one part
    view = df.iloc[max(0, cursor - steps + 1):cursor + 1]
    phys_parts = [r["_key"] for r in rows if r["_phys"]]
    default = min(phys_parts, key=lambda k: np.nan_to_num(cur.get(f"cb_{k}"), nan=100))
    # the worst part is only the starting choice; after that the user's choice is kept
    # (the selectbox has a fixed key, so it no longer jumps when the worst part changes during replay)
    sel_key = f"hyb_part_{ds}"
    if st.session_state.get(sel_key) not in phys_parts:
        st.session_state[sel_key] = default
    part = st.selectbox("Show the details for", phys_parts, key=sel_key,
                        format_func=lambda k: PART_NAMES[k])
    sensors = [t for t in pt.PHYS_PARTS[part] if t in p["targets"]]
    colours = [BLUE, RED, PURPLE]
    win = view.loc[view["in_event_window"].fillna(False).astype(bool), "time_stamp"] if show_window else []

    def shade(fig):
        # always show the legend (plotly hides it when there is only one line) and put it
        # under the chart, so long names are not cut off by the title
        # own uirevision per dataset + part + chart: switching part or dataset starts the chart fresh
        # instead of reusing the old chart's lines / hidden-legend state
        fig.update_layout(uirevision=f"hyb-{ds}-{part}-{fig.layout.title.text}")
        fig.update_layout(showlegend=True, margin=dict(l=10, r=10, t=46, b=70),
                          legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0))
        if len(win):
            fig.add_vrect(x0=win.min(), x1=win.max(), fillcolor=RED, opacity=0.07, line_width=0,
                          annotation_text="fault window", annotation_position="top right",
                          annotation_font=dict(size=10, color=RED))
        return fig

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for t, colr in zip(sensors, colours):
            band = p["targets"][t]["band"]
            fig.add_trace(go.Scatter(x=view["time_stamp"], y=view[f"{t}_gap_day"], name=pt.TARGETS[t][0],
                                     line=dict(color=colr, width=2)))
            fig.add_hline(y=band, line=dict(color=colr, dash="dot", width=1))
        fig.add_hline(y=0, line=dict(color=GREY, width=1))
        st.plotly_chart(shade(style(fig, "Physics: measured − expected temperature (daily mean)", 380, "°C")),
                        use_container_width=True)
        st.caption("Dotted lines = normal limit for each sensor (99 % of healthy days stay below it).")
    with c2:
        thr = float(cur.get("threshold", np.nan))
        fig = go.Figure()
        for s, colr in zip([s for s in PARTS[part] if s in view], [TEAL, AMBER, PURPLE]):
            fig.add_trace(go.Scatter(x=view["time_stamp"], y=view[s], name=s, line=dict(color=colr, width=2)))
        fig.add_hline(y=thr, line=dict(color=INK, dash="dash", width=1), annotation_text=f"alarm level {thr:.0f}",
                      annotation_position="top left", annotation_font=dict(size=10, color=MUTED))
        st.plotly_chart(shade(style(fig, "Learning model: anomaly score", 380, "score")), use_container_width=True)
        st.caption("Above the alarm level for 3 h = learning-model alarm.")
    ws = pt.worst_sensor(cur, part, p)
    t = ws[0] if ws else sensors[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view["time_stamp"], y=view[f"{t}_meas"], name="Measured", line=dict(color=TEAL, width=1.6)))
    fig.add_trace(go.Scatter(x=view["time_stamp"], y=view[f"{t}_exp"], name="Physics expects",
                             line=dict(color=BLUE, width=1.6, dash="dash")))
    st.plotly_chart(shade(style(fig, f"{pt.TARGETS[t][0].capitalize()}: measured vs what physics expects", 380, "°C")),
                    use_container_width=True)

    with st.expander("How the physics model was tuned for this turbine"):
        st.markdown(f"Turbine **{p['asset']}**, tuned on its most recent healthy data "
                    f"**{p['cal_from'][:10]} → {p['cal_to'][:10]}** ({p['cal_rows']:,} ten-minute steps). "
                    "Test periods and fault windows of this turbine were left out. Same equations and method as "
                    "`calibrate_twin.m` in the MATLAB folder. Power curve from physics: "
                    f"R² {p['P_r2']:.3f}, rotor radius ≈ {p['R']:.0f} m.")
        st.dataframe(pd.DataFrame([{"Sensor": pt.TARGETS[k][0], "Heats up in (h)": q["tau_h"], "R²": round(q["r2"], 3),
                                    "Typical error (°C)": round(q["rmse"], 1), "Normal daily limit (°C)": round(q["band"], 1)}
                                   for k, q in p["targets"].items()]), use_container_width=True, hide_index=True)
        st.caption("Combined verdict: green = both normal · amber = one model worried · red = both models agree.")


def page_3d(info, ds, data, cursor, steps, model, synthetic=False):
    section_title("3D Twin", "Rotor speed, blade pitch and wind come from the data; every part is coloured by its health "
                             "from the fault detection (green healthy · amber watch · red critical)")
    if twin3d is None:
        st.error("twin3d.py not found - put it in the same folder as digital_twin_app.py.")
        return
    rated = model["rated_power"] if model else 1.0
    modes = ["Both models (combined)", "Learning model", "Physics model"]
    mode = modes[1] if synthetic else st.radio("Colour the parts by", modes, horizontal=True, key="col3d")
    health_cols = None
    if mode != modes[1]:
        df, p, msg = with_physics(info, ds, data)
        if p is None:
            st.info(msg + " Showing the learning model.")
        else:
            data = df
            health_cols = ({k: f"cb_{k}" for k in PARTS} if mode == modes[0]
                           else {k: f"phh_{k}" for k in PARTS})
    tl = twin3d.build_timeline(data, cursor, steps, rated, PARTS, health_cols=health_cols)
    title = info.get(ds, {}).get("text", str(ds)) + ("" if health_cols is None else f" · colours: {mode.lower()}")
    components.html(twin3d.turbine3d_html(tl, title, autoplay=True, height=660), height=672, scrolling=False)
    tip = ("Live synthetic turbine: inject a fault in the sidebar, press **+1 day**, and the 3D view replays the new data."
           if synthetic else "Plays the **chart window** chosen in the sidebar, ending at the current twin time.")
    st.caption(f"{tip} Drag to rotate, scroll to zoom, **View** for camera presets, **X-ray** to see inside the nacelle. "
               "The grey background turbines are decoration; only the front turbine follows the data. In *physics model* colours, parts without a physics model are grey. Works offline.")


def main():
    st.set_page_config(page_title="Wind Turbine Digital Twin", page_icon="🌬️", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.sidebar.markdown(BRAND, unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")

    model = load_power_model()
    sigma = power_error_sigma()
    fs = load_json(os.path.join(FAULT_DIR, "summary.json")) or {}
    crit_limit = int(fs.get("settings", {}).get("crit_limit", 72))
    info = dataset_list()
    if not info:
        st.error("No fault detection scores found. Run `python fault_detection.py` first.")
        return
    sources = ["Recorded data (replay)"] + (["Live synthetic turbine"] if syn is not None and fd is not None else [])
    synthetic = st.sidebar.radio("Data source", sources, key="source") == "Live synthetic turbine"
    if synthetic:
        tb, base, steps, show_window, playing, speed = syn_controls(info, model, crit_limit)
        ds, data, pos_key = SYN, tb.frame(), None
    else:
        ds, data, pos_key, speed, steps, show_window, playing = sidebar_controls(info)
    st.sidebar.markdown(f"<div style='margin-top:18px;color:{MUTED};font-size:12px'>🌱 Clean energy · data: CARE to Compare, Wind Farm A</div>",
                        unsafe_allow_html=True)
    if model is None:
        st.sidebar.warning("Power model not found - run power_model.py.")

    live = page in LIVE_PAGES
    n = len(data)

    def body():
        """Page content. During replay only this part re-runs (Streamlit fragment), so the sidebar
        and the page layout stay in place and the charts update instead of the whole app reloading."""
        nonlocal_info = info
        if synthetic:
            if playing and live:
                tb.step(speed)
            data_now = tb.frame()
            cursor = len(data_now) - 1
            faults = " + ".join(f"{f['name']} ({f['severity'].lower()})" for f in tb.faults) or "no fault injected"
            nonlocal_info = dict(info)
            nonlocal_info[SYN] = {"asset": "SYN", "label": "synthetic", "fault": faults,
                                  "text": f"SYNTHETIC turbine imitating dataset {base} · weather: {tb.scenario} · {faults}"}
        else:
            data_now = data
            if playing and live:
                st.session_state[pos_key] = min(n - 1, st.session_state[pos_key] + speed)
            cursor = int(st.session_state[pos_key])
        if page == PAGES[0]:
            page_overview(nonlocal_info, ds, data_now, cursor, steps, show_window, model, sigma, crit_limit)
        elif page == PAGES[1]:
            page_power(model, data_now, cursor, sigma)
        elif page == PAGES[2]:
            page_anomaly(nonlocal_info, ds, data_now, cursor, steps, show_window, sigma)
        elif page == PAGES[3]:
            page_health(ds, data_now, cursor, model)
        elif page == PAGES[4]:
            page_hybrid(nonlocal_info, ds, data_now, cursor, steps, show_window)
        elif page == PAGES[5]:
            page_3d(nonlocal_info, ds, data_now, cursor, steps, model, synthetic)
        else:
            page_reports()
        if playing and live and not synthetic and cursor >= n - 1:
            st.info("End of the data - switch Play off and drag Position to replay again.")

    fragment = getattr(st, "fragment", None)          # Streamlit >= 1.37
    if fragment is not None:
        fragment(run_every=FRAME_SECONDS if (playing and live) else None)(body)()
    else:                                              # older Streamlit: rerun the whole app
        body()
        if playing and live and (synthetic or st.session_state[pos_key] < n - 1):
            time.sleep(FRAME_SECONDS)
            st.rerun()


if __name__ == "__main__":
    main()
