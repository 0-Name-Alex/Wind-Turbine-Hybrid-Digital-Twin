function build_twin_model(mdl)
%BUILD_TWIN_MODEL  Creates the Simulink model wt_twin.slx automatically.
%   build_twin_model            % builds and opens wt_twin.slx
%
%   Layout:
%     From Workspace (wind, nacelle temp, ambient temp, fault severity)
%       -> Aero_Control  (Cp-lambda rotor, MPPT + pitch control)   [wt_aero.m]
%       -> Losses        (losses -> equilibrium temperatures, faults) [wt_heat.m]
%       -> Thermal       (first-order heat lag, one per component)
%       -> To Workspace  (P_sim, w_sim, beta_sim, T_sim)
%   Parameters come from base-workspace variables set by run_twin.m:
%     aero_par, k_fault, C_therm, a_therm, T0, use_meas (+ load_in)
%   Plain Simulink only (no Simscape). Fixed-step discrete, Ts = 600 s.

if nargin < 1, mdl = 'wt_twin'; end
if bdIsLoaded(mdl), close_system(mdl, 0); end
if exist([mdl '.slx'], 'file'), delete([mdl '.slx']); end

new_system(mdl);
open_system(mdl);
set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', ...
               'FixedStep', '600', 'StopTime', '86400');

%% ---------------- inputs (From Workspace) and parameters (Constant)
fw = @(name, var, y) add_block('simulink/Sources/From Workspace', [mdl '/' name], ...
        'VariableName', var, 'SampleTime', '600', 'Interpolate', 'off', ...
        'OutputAfterFinalValue', 'Holding final value', 'Position', [30 y 130 y+30]);
cn = @(name, var, y) add_block('simulink/Sources/Constant', [mdl '/' name], ...
        'Value', var, 'Position', [30 y 130 y+30]);

fw('Wind',      'wind_in',  40);
fw('Faults',    'fault_in', 110);
cn('AeroPar',   'aero_par', 180);
cn('FaultGain', 'k_fault',  250);
fw('T_nacelle', 'tnac_in',  320);
fw('T_ambient', 'tamb_in',  390);
cn('ThermCoef', 'C_therm',  460);
fw('Load_meas', 'load_in',  530);     % measured [P w] for measured-load mode
cn('UseMeas',   'use_meas', 600);     % 1 = temperatures from measured load

%% ---------------- MATLAB Function blocks
mf = 'simulink/User-Defined Functions/MATLAB Function';
add_block(mf, [mdl '/Aero_Control'], 'Position', [230 40 370 280]);
add_block(mf, [mdl '/Losses'],       'Position', [450 200 590 620]);
rt = sfroot;
set_script(rt, [mdl '/Aero_Control'], strjoin({ ...
    'function [P, w_pu, beta] = aero(v, f, ap, kf)'
    '% Rotor aerodynamics + MPPT / pitch control (see wt_aero.m)'
    '% ap = [Kp R wmin Prated Pstandby], hydraulic fault = pitch offset kf(4)*f(3)'
    '[P, w_pu, beta] = wt_aero(v, kf(4)*f(3), ap(1), ap(2), ap(3), ap(4), ap(5));'}', newline));
set_script(rt, [mdl '/Losses'], strjoin({ ...
    'function [P_grid, Teq] = losses(P, w_pu, Tnac, Tamb, f, C, kf, Lm, um)'
    '% Component losses -> equilibrium temperatures, with fault injection (see wt_heat.m)'
    '% um = 1: heat from MEASURED power/speed (fault detection, ignores curtailment)'
    '% um = 0: heat from SIMULATED power/speed (pure physics, what-if scenarios)'
    'Pl = P; wl = w_pu;'
    'if um > 0.5, Pl = Lm(1); wl = Lm(2); end'
    '[~, Teq] = wt_heat(Pl, wl, Tnac, Tamb, f, C, kf);'
    'P_grid = P*(1 - kf(2)*f(1));'}', newline));

%% ---------------- thermal subsystem: T[k] = T[k-1] + a*(Teq[k] - T[k-1])
th = [mdl '/Thermal'];
add_block('built-in/Subsystem', th, 'Position', [670 315 790 375]);
add_block('built-in/Inport',  [th '/Teq'],  'Position', [20 50 50 70]);
add_block('built-in/Sum',     [th '/Err'],  'Inputs', '+-', 'Position', [100 45 130 75]);
add_block('built-in/Gain',    [th '/a'],    'Gain', 'a_therm', ...
          'Multiplication', 'Element-wise(K.*u)', 'Position', [170 45 230 75]);
add_block('built-in/Sum',     [th '/Next'], 'Inputs', '++', 'Position', [280 40 310 80]);
add_block('built-in/UnitDelay', [th '/T_prev'], 'InitialCondition', 'T0', ...
          'SampleTime', '600', 'Orientation', 'left', 'Position', [180 130 220 170]);
add_block('built-in/Outport', [th '/T'],    'Position', [370 50 400 70]);
add_line(th, 'Teq/1',    'Err/1',    'autorouting', 'on');
add_line(th, 'T_prev/1', 'Err/2',    'autorouting', 'on');
add_line(th, 'Err/1',    'a/1',      'autorouting', 'on');
add_line(th, 'T_prev/1', 'Next/1',   'autorouting', 'on');
add_line(th, 'a/1',      'Next/2',   'autorouting', 'on');
add_line(th, 'Next/1',   'T_prev/1', 'autorouting', 'on');
add_line(th, 'Next/1',   'T/1',      'autorouting', 'on');

%% ---------------- outputs (To Workspace)
tw = @(name, var, pos) add_block('simulink/Sinks/To Workspace', [mdl '/' name], ...
        'VariableName', var, 'SaveFormat', 'Array', 'SampleTime', '600', ...
        'MaxDataPoints', 'inf', 'Position', pos);
tw('w_out',    'w_sim',    [450 60 530 90]);
tw('beta_out', 'beta_sim', [450 120 530 150]);
tw('P_out',    'P_sim',    [870 230 950 260]);
tw('T_out',    'T_sim',    [870 330 950 360]);

%% ---------------- wiring
L = @(a, b) add_line(mdl, a, b, 'autorouting', 'on');
L('Wind/1',      'Aero_Control/1');
L('Faults/1',    'Aero_Control/2');
L('AeroPar/1',   'Aero_Control/3');
L('FaultGain/1', 'Aero_Control/4');
L('Aero_Control/1', 'Losses/1');
L('Aero_Control/2', 'Losses/2');
L('T_nacelle/1', 'Losses/3');
L('T_ambient/1', 'Losses/4');
L('Faults/1',    'Losses/5');
L('ThermCoef/1', 'Losses/6');
L('FaultGain/1', 'Losses/7');
L('Load_meas/1', 'Losses/8');
L('UseMeas/1',   'Losses/9');
L('Aero_Control/2', 'w_out/1');
L('Aero_Control/3', 'beta_out/1');
L('Losses/1',    'P_out/1');
L('Losses/2',    'Thermal/1');
L('Thermal/1',   'T_out/1');

save_system(mdl);
fprintf('Built %s.slx\n', mdl);
end

function set_script(rt, path, code)
ch = rt.find('-isa', 'Stateflow.EMChart', 'Path', path);
ch.Script = code;
end
