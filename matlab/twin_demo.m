



%% TWIN_DEMO  -  One script for the whole MATLAB physics twin
%  ------------------------------------------------------------------------
%  What it does
%    1. Checks the files, tunes the twin (calibrate_twin) and builds the
%       Simulink model (build_twin_model) if they are missing.
%    2. Runs the HEALTHY twin on real wind / temperature data.
%    3. Runs the SAME period again with the fault(s) you choose.
%    4. Prints accuracy (twin vs real SCADA) and the fault effect per part:
%       temperature rise, energy loss, and whether it is DETECTABLE
%       (bigger than the normal day-to-day error of the healthy twin).
%    5. Draws the graphs and saves everything in matlab_results\my_run_<time>
%
%  How to use
%    * Put this file in D:\Projects\RET_matlab (next to run_twin.m etc.)
%    * Set the Current Folder to that folder.
%    * Either edit the SETTINGS below, or set ASK = true to pick everything
%      from pop-up menus.  Then press Run (F5) or type  twin_demo
%  ------------------------------------------------------------------------

clear; close all; clc;

%% ============================ SETTINGS ===================================
ASK        = true;            % true = pop-up menus, false = use the values below

ASSET      = 10;              % turbine (twin_params_A10.mat must match)
CSV_FILE   = 'matlab_data/A10_ev03_healthy.csv';   % real weather/data to drive the twin
DAYS       = 30;              % how many days to simulate
START_DATE = [];              % [] = start of file, or e.g. datetime(2023,4,27)
LOAD_MODE  = 'simulated';     % 'simulated' = everything from wind (best for faults)
                              % 'measured'  = temperatures from real power (best for accuracy)

% ---- faults: one row per fault.  Leave FAULTS = {} for a healthy run only
%   type      : 'gearbox' | 'gen_bearing' | 'hydraulic' | 'transformer'
%   severity  : 'Mild' | 'Moderate' | 'Severe'   (or a number 0..1)
%   start_day : day the fault starts (counted from the first simulated day)
%   ramp_days : 0 = sudden, 1 or 3 = grows over that many days
FAULTS = { ...
%   type          severity    start_day  ramp_days
    'gearbox',    'Severe',   10,        3 ;
};

OPEN_SIMULINK = false;        % true = open wt_twin.slx at the end (press Run there)
%% ========================================================================

%% ---- 0. pop-up menus (optional) -----------------------------------------
types    = {'gearbox', 'gen_bearing', 'hydraulic', 'transformer'};
typeName = {'Gearbox (more friction loss, lower efficiency)', ...
            'Generator bearing (more friction)', ...
            'Hydraulic pitch (blades misaligned 3 deg at severe)', ...
            'Transformer (more copper loss)'};
sevName  = {'Mild', 'Moderate', 'Severe'};
if ASK
    files = dir(fullfile('matlab_data', '*.csv'));
    if ~isempty(files)
        i0 = find(strcmp({files.name}, 'A10_ev03_healthy.csv'), 1); if isempty(i0), i0 = 1; end
        [i, ok] = listdlg('PromptString', 'Data to drive the twin:', 'SelectionMode', 'single', ...
                          'ListString', {files.name}, 'ListSize', [320 300], 'InitialValue', i0);
        if ~ok, return; end
        CSV_FILE = fullfile('matlab_data', files(i).name);
        tok = regexp(files(i).name, '^A(\d+)_', 'tokens', 'once');
        if ~isempty(tok), ASSET = str2double(tok{1}); end
    end
    [i, ok] = listdlg('PromptString', 'Fault(s) to inject (Ctrl+click for several, Cancel = none):', ...
                      'ListString', typeName, 'ListSize', [380 120]);
    FAULTS = {};
    if ok
        for k = i
            [s, ok2] = listdlg('PromptString', ['Severity: ' typeName{k}], 'SelectionMode', 'single', ...
                               'ListString', sevName, 'InitialValue', 3, 'ListSize', [300 80]);
            if ~ok2, s = 3; end
            a = inputdlg({'Start day', 'Grows over how many days (0 = sudden)'}, types{k}, 1, {'10', '3'});
            if isempty(a), a = {'10', '3'}; end
            FAULTS(end + 1, :) = {types{k}, sevName{s}, str2double(a{1}), str2double(a{2})}; %#ok<SAGROW>
        end
    end
    a = inputdlg({'Days to simulate', 'Load mode (simulated / measured)'}, 'Run settings', 1, ...
                 {num2str(DAYS), LOAD_MODE});
    if ~isempty(a), DAYS = str2double(a{1}); LOAD_MODE = strtrim(a{2}); end
end

%% ---- 1. checks, tuning, model -------------------------------------------
PARAMS = sprintf('twin_params_A%02d.mat', ASSET);
assert(exist(CSV_FILE, 'file') == 2, 'Data file not found: %s (run export_for_matlab.py first)', CSV_FILE);
if ~exist(PARAMS, 'file')
    fprintf('Tuning the twin for turbine %d (first time only)...\n', ASSET);
    calibrate_twin(ASSET);
end
if ~exist('wt_twin.slx', 'file'), build_twin_model; end
S = load(PARAMS); p = S.p;
nm   = p.names;                                     % the 5 modelled temperatures
pretty = {'Gearbox oil', 'Gearbox HSS bearing', 'Generator bearing (DE)', 'Stator winding', 'HV transformer'};

% build the fault struct
f = struct('type', {}, 'severity', {}, 'start_day', {}, 'ramp_days', {});
for k = 1:size(FAULTS, 1)
    sv = FAULTS{k, 2};
    if ischar(sv) || isstring(sv), sv = find(strcmpi(sevName, sv)) / 3; end
    assert(~isempty(sv) && any(strcmp(types, FAULTS{k, 1})), 'Bad fault row %d', k);
    f(end + 1) = struct('type', FAULTS{k, 1}, 'severity', sv, ...
                        'start_day', FAULTS{k, 3}, 'ramp_days', FAULTS{k, 4}); %#ok<SAGROW>
end
opt = {'Days', DAYS, 'Plot', false, 'Load', LOAD_MODE};
if ~isempty(START_DATE), opt = [opt, {'StartDate', START_DATE}]; end

fprintf('\n=== Physics twin run ===\nData: %s | turbine %d | %d days | load mode: %s\n', ...
        CSV_FILE, ASSET, DAYS, LOAD_MODE);
for k = 1:numel(f)
    fprintf('Fault %d: %-12s severity %.2f  start day %g  ramp %g d\n', k, f(k).type, ...
            f(k).severity, f(k).start_day, f(k).ramp_days);
end

%% ---- 2. healthy run (also prints accuracy vs real SCADA) ----------------
H = run_twin(CSV_FILE, PARAMS, [], opt{:});

%% ---- 3. faulty run -------------------------------------------------------
if isempty(f)
    F = H;  fprintf('\nNo fault chosen - healthy run only.\n');
else
    F = run_twin(CSV_FILE, PARAMS, f, opt{:});
end

%% ---- 4. fault effect + detectable? --------------------------------------
t0     = H.time(1);
fstart = min([f.start_day, inf]);
last   = H.time >= H.time(end) - days(5);           % last 5 days
after  = H.time >= t0 + days(fstart);
dT = zeros(1, 5); det = false(1, 5);
for k = 1:5
    dT(k)  = mean(F.([nm{k} '_sim'])(last) - H.([nm{k} '_sim'])(last), 'omitnan');
    det(k) = dT(k) > p.res_day_p99(k);
end
e0 = sum(max(H.P_sim(after), 0)); e1 = sum(max(F.P_sim(after), 0));
lossPct = 100 * (e0 - e1) / max(e0, eps);
dP  = mean(H.P_sim(last) - F.P_sim(last));
detP = dP > abs(p.P_day_p1);

fprintf('\n=== Fault effect (mean of the last 5 days, faulty minus healthy twin) ===\n');
fprintf('%-24s %9s %12s  %s\n', 'Part', 'Rise (C)', 'Normal band', 'Detectable?');
for k = 1:5
    fprintf('%-24s %+9.2f %10.1f C  %s\n', pretty{k}, dT(k), p.res_day_p99(k), yesno(det(k)));
end
fprintf('%-24s %8.1f %%  %9.3f pu  %s\n', 'Energy lost', lossPct, abs(p.P_day_p1), yesno(detP));
if isempty(f)
    verdict = 'healthy run';
elseif any(det) || detP
    verdict = 'DETECTABLE - the fault rises above normal variation';
else
    verdict = 'NOT detectable - hidden inside normal variation of 10-min data';
end
fprintf('\nVerdict: %s\n', verdict);

%% ---- 5. graphs -----------------------------------------------------------
outDir = fullfile('matlab_results', ['my_run_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))]);
mkdir(outDir);
fs = H.time(1) + days(fstart);

% (a) twin vs real SCADA (healthy twin) - shows the model is right
fig1 = figure('Name', 'Twin vs SCADA', 'Color', 'w', 'Position', [60 60 1100 700]);
subplot(3, 1, 1); plot(H.time, H.P_meas, '.', 'MarkerSize', 3); hold on; plot(H.time, H.P_sim, 'LineWidth', 1);
ylabel('Power (pu)'); legend('Measured', 'Healthy twin'); grid on; title('Healthy twin vs real SCADA');
subplot(3, 1, 2); plot(H.time, H.([nm{2} '_meas']), '.', 'MarkerSize', 3); hold on;
plot(H.time, H.([nm{2} '_sim']), 'LineWidth', 1); ylabel('\circC'); grid on;
legend('Measured', 'Twin'); title(pretty{2});
subplot(3, 1, 3); plot(H.time, H.([nm{4} '_meas']), '.', 'MarkerSize', 3); hold on;
plot(H.time, H.([nm{4} '_sim']), 'LineWidth', 1); ylabel('\circC'); grid on;
legend('Measured', 'Twin'); title(pretty{4});
exportgraphics(fig1, fullfile(outDir, 'twin_vs_scada.png'), 'Resolution', 200);

% (b) healthy vs faulty twin for every part + power
if ~isempty(f)
    fig2 = figure('Name', 'Fault injection', 'Color', 'w', 'Position', [80 40 1200 800]);
    for k = 1:5
        subplot(3, 2, k);
        plot(H.time, H.([nm{k} '_sim']), 'LineWidth', 1); hold on;
        plot(F.time, F.([nm{k} '_sim']), 'LineWidth', 1);
        plot(H.time, H.([nm{k} '_sim']) + p.res_day_p99(k), 'k:');
        xline(fs, 'r--'); grid on; ylabel('\circC');
        title(sprintf('%s  (rise %+.1f C, %s)', pretty{k}, dT(k), yesno(det(k))));
        if k == 1, legend('Healthy twin', 'With fault', 'Normal band', 'Fault start', 'Location', 'best'); end
    end
    subplot(3, 2, 6);
    plot(H.time, H.P_sim, 'LineWidth', 1); hold on; plot(F.time, F.P_sim, 'LineWidth', 1);
    xline(fs, 'r--'); grid on; ylabel('Power (pu)');
    title(sprintf('Power  (energy lost %.1f %%)', lossPct));
    sgtitle(sprintf('Injected: %s', strjoin(arrayfun(@(x) sprintf('%s %.2f', x.type, x.severity), f, ...
            'UniformOutput', false), ', ')));
    exportgraphics(fig2, fullfile(outDir, 'fault_injection.png'), 'Resolution', 200);
end

%% ---- 6. save results ------------------------------------------------------
writetable(H, fullfile(outDir, 'healthy_run.csv'));
if ~isempty(f), writetable(F, fullfile(outDir, 'faulty_run.csv')); end
Summary = table(string(pretty'), dT', p.res_day_p99(:), det', ...
                'VariableNames', {'Part', 'Rise_C', 'NormalBand_C', 'Detectable'});
writetable(Summary, fullfile(outDir, 'summary.csv'));
fprintf('\nSaved graphs and tables in %s\n', outDir);

if OPEN_SIMULINK, open_system('wt_twin'); end

%% ---- helper ----------------------------------------------------------------
function s = yesno(b)
if b, s = 'YES'; else, s = 'no'; end
end
