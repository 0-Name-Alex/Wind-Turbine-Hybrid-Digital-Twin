# Physics twin (MATLAB / Simulink) – Part 1 + Part 2

Plain MATLAB + Simulink only. No Simscape and no toolboxes are needed.

## Files
| File | What it does |
|---|---|
| `wt_cp.m` | Blade efficiency Cp(λ, β) (Heier model) |
| `wt_aero.m` | Wind → power, rotor speed, pitch (MPPT below rated, pitch control above) |
| `wt_power_vec.m` | Fast version of `wt_aero` used for calibration |
| `wt_heat.m` | Power and speed → losses → equilibrium temperature of 5 parts, with fault injection |
| `load_scada.m` | Reads a CSV from `export_for_matlab.py`, removes the 205 °C error code |
| `calibrate_twin.m` | Tunes the twin to one turbine's healthy data → `twin_params_A10.mat` |
| `build_twin_model.m` | Builds the Simulink model `wt_twin.slx` automatically |
| `run_twin.m` | Runs the Simulink model on a SCADA file, compares with the measured values, plots |
| `run_fault_scenarios.m` | Part 2: 36 injected-fault runs → `matlab_results/scenario_summary.csv` |

The 5 modelled temperatures are: gearbox oil, gearbox high-speed-shaft bearing, generator bearing (drive end), stator winding, and HV transformer.

## Steps
1. Copy these files into `D:\Projects\RET -- Claude` (next to the `matlab_data` folder).
2. In MATLAB, set the Current Folder to that folder.
3. Run these in the Command Window:

```matlab
p = calibrate_twin(10);          % Part 1: tune to turbine 10's healthy data
build_twin_model;                % creates and opens wt_twin.slx

% check the twin against real data (pure physics: everything from wind)
R = run_twin('matlab_data/A10_ev03_healthy.csv', 'twin_params_A10.mat', [], ...
             'StartDate', datetime(2023,4,27), 'Days', 30);

% the same check, but temperatures are driven by the measured power
% (this is the mode to use for fault detection)
R = run_twin('matlab_data/A10_ev03_healthy.csv', 'twin_params_A10.mat', [], ...
             'Days', 60, 'Load', 'measured');

T = run_fault_scenarios;         % Part 2: break it on purpose
```

To calibrate another turbine, use `calibrate_twin(0)`, `(11)`, `(13)` or `(21)`. This needs that turbine's healthy CSVs in `matlab_data`.

## What to expect
These are results from a Python copy of the same equations, run on turbine 10. MATLAB should match closely.

| | R² | Error |
|---|---|---|
| Power, pure physics (healthy test period) | 0.988 | nRMSE 3.5 % |
| Gearbox HSS bearing, measured load | 0.96 | 2.0 °C |
| Stator winding, measured load | 0.99 | 2.7–3.0 °C |
| Gearbox oil, measured load | 0.85 | 2.6–2.8 °C |
| Transformer, measured load | 0.6–0.85 | 4–5 °C |
| Generator bearing, measured load | 0.6–0.85 | 5–6 °C |

Fitted values: rotor radius ≈ 43 m, rated wind speed ≈ 12.4 m/s, rated power 0.975 pu.

The thermal time constants come out at 0.5 h for the gearbox bearing, 1 h for the generator, stator and transformer, and 3 h for the gearbox oil.

## Why two load modes
The operator sometimes curtails the turbine. In May 2022 and January 2023 this happened about 10 % of the time, and the status flag still said "normal". Physics cannot predict an operator's decision, so:

- **`'simulated'`** is the pure twin. Use it for what-if fault scenarios.
- **`'measured'`** drives the temperatures from the measured power. Use it for fault detection.

## Fault strengths at Severe
These are engineering assumptions; state them as assumptions in the report. They are set in `p.kfault` inside `calibrate_twin.m`.

| Fault | What changes at Severe | Expected 5-day temperature or power change |
|---|---|---|
| Gearbox | +50 % gear and bearing losses, −2 % efficiency | +6 °C on the gearbox |
| Generator bearing | friction doubled | +8 °C on the bearing |
| Hydraulic pitch | 3° blade misalignment | about 23 % less energy, no heat |
| Transformer | +60 % winding losses | +0.7 °C in a low-wind month; much more at high load |

Mild faults are 1/3 of these, and Moderate faults are 2/3.

## Main findings so far (turbine 10)
1. **Mild faults of every type stay inside the normal healthy variation.** This uses the healthy daily error band (99th percentile): gearbox about 3.5–4.3 °C, stator 4.8 °C, transformer 7.6 °C, generator bearing 9 °C. Only moderate and severe gearbox faults clearly stand out. This is a physical explanation for why 8 of the 12 real faults never exceed healthy levels.
2. **Hydraulic faults show up as lost power, not heat.** Curtailment causes power drops of a similar size, so in 10-minute SCADA a hydraulic fault looks like normal curtailment. Six of the 12 faults are hydraulic.
3. **The real gearbox failure (event 10) leaves no heat signature** in the physics residuals: the mean residual in the fault window is −1.0 °C for the oil and +0.2 °C for the bearing.
4. **Generator bearing on turbine 10:** the bearing ran about 3 °C hotter than the twin in mid-2022, then about 5 °C cooler from April 2023. Its bearing-failure event 40 (Dec 2022 – Jan 2023) falls in between. This suggests the twin saw the worn bearing early and the repaired one afterwards. Confirm it with event 40's CSV.

   Because of this, turbine 10's healthy calibration year includes the worn-bearing months, which widens the generator-bearing band (9 °C).

## Known limits
- 10-minute averages, so there are no fast dynamics such as drivetrain oscillation.
- The turbine specs are anonymised, so the parameters are calibrated rather than taken from a datasheet.
- Fault strengths are assumptions, not measured values.
