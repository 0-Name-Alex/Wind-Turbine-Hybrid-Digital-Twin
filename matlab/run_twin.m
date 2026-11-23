function R = run_twin(csvFile, paramsFile, faults, varargin)
%RUN_TWIN  Run the Simulink twin on one SCADA file (with optional faults).
%   R = run_twin('matlab_data/A10_ev03_healthy.csv', 'twin_params_A10.mat');
%   R = run_twin(csv, params, [], 'StartDate', datetime(2022,6,1), 'Days', 30);
%
%   faults = struct array, e.g.
%     faults = struct('type', 'gearbox', 'severity', 1, 'start_day', 10, 'ramp_days', 3);
%   type      : 'gearbox' | 'gen_bearing' | 'hydraulic' | 'transformer'
%   severity  : 0..1   (Mild 1/3, Moderate 2/3, Severe 1)
%   start_day : day the fault starts (from the first simulated sample)
%   ramp_days : 0 = sudden, otherwise grows linearly over this many days
%
%   Options: 'StartDate', 'Days', 'Plot' (true/false), 'SaveCSV' (file name),
%            'Load' : 'simulated' -> temperatures from simulated power (pure twin)
%                     'measured'  -> temperatures from measured power/speed
%                                    (use for fault detection: curtailment is
%                                     an operator decision physics cannot know)
%   Output table R: measured vs simulated power and temperatures, residuals,
%   fault severity used, and the evaluation-only columns (status, in_event).

ip = inputParser;
addParameter(ip, 'StartDate', []);
addParameter(ip, 'Days', []);
addParameter(ip, 'Plot', true);
addParameter(ip, 'SaveCSV', '');
addParameter(ip, 'Load', 'simulated');   % 'simulated' (pure physics) or 'measured'
parse(ip, varargin{:});
o = ip.Results;
if nargin < 3, faults = []; end

mdl = 'wt_twin';
if ~exist([mdl '.slx'], 'file'), build_twin_model(mdl); end
load_system(mdl);

S = load(paramsFile); p = S.p;
D = load_scada(csvFile);
[~, iu] = unique(D.time); D = D(iu, :);
if ~isempty(o.StartDate), D = D(D.time >= o.StartDate, :); end
if ~isempty(o.Days),      D = D(D.time <  D.time(1) + days(o.Days), :); end

%% ---- uniform 10-min grid (Simulink needs equally spaced inputs)
% Index every sample by its 10-min slot number (exact integers, so no
% floating-point mismatch between the grid and the measured timestamps)
t0  = dateshift(D.time(1), 'start', 'minute');
idx = round(minutes(D.time - t0) / 10) + 1;
Ng  = max(idx);
tg  = t0 + minutes(10 * (0:Ng-1))';
hit = false(Ng, 1); loc = zeros(Ng, 1);
hit(idx) = true;    loc(idx) = (1:height(D))';
grid = @(x) fill_grid(x, hit, loc);
wind = grid(D.wind); tnac = grid(D.T_nacelle); tamb = grid(D.T_amb);
pm = grid(D.P); wm = grid(D.w);
ts = seconds(tg - tg(1));
td = ts / 86400;

%% ---- inputs and parameters for the model (base workspace)
F = fault_profile(td, faults);
nm = p.names;
T0 = zeros(5, 1);
for k = 1:5
    x = D.(nm{k}); x0 = x(find(~isnan(x), 1));
    if isempty(x0), x0 = tnac(1) + 10; end
    T0(k) = x0;
end
assignin('base', 'wind_in',  [ts wind]);
assignin('base', 'tnac_in',  [ts tnac]);
assignin('base', 'tamb_in',  [ts tamb]);
assignin('base', 'fault_in', [ts F]);
assignin('base', 'load_in',  [ts pm wm]);
assignin('base', 'use_meas', double(strcmpi(o.Load, 'measured')));
assignin('base', 'aero_par', [p.Kp p.R p.wmin p.Prated p.Pstandby]);
assignin('base', 'k_fault',  p.kfault);
assignin('base', 'C_therm',  p.C);
assignin('base', 'a_therm',  600 ./ (p.tau_h(:) * 3600));
assignin('base', 'T0',       T0);

for b = {'P_out', 'T_out', 'w_out', 'beta_out'}       % keep every sample
    set_param([mdl '/' b{1}], 'MaxDataPoints', 'inf', 'Decimation', '1');
end
stopT = sprintf('%d', round(ts(end)));
set_param(mdl, 'StopTime', stopT);
out = sim(mdl, 'StopTime', stopT, 'ReturnWorkspaceOutputs', 'on');
Ps = as_rows(out.get('P_sim')); Ts = as_rows(out.get('T_sim'));
ws = as_rows(out.get('w_sim')); bs = as_rows(out.get('beta_sim'));
n  = min([numel(tg), size(Ps, 1), size(Ts, 1)]);
fprintf('\nGrid: %d steps (%.1f days) | measured samples matched: %d | Simulink returned %d x %d temperatures\n', ...
        numel(tg), td(end), nnz(hit), size(Ts, 1), size(Ts, 2));
if n < 0.9 * numel(tg)
    warning('Simulink returned fewer samples than expected - send this message to Claude.');
end

%% ---- results table
R = table(tg(1:n), wind(1:n), 'VariableNames', {'time', 'wind'});
R.P_meas = nan(n, 1); R.P_meas(hit(1:n)) = D.P(loc(hit(1:n)));
R.P_sim = Ps(1:n); R.w_sim = ws(1:n); R.beta_sim = bs(1:n);
for k = 1:5
    m = nan(n, 1); m(hit(1:n)) = D.(nm{k})(loc(hit(1:n)));
    R.([nm{k} '_meas']) = m;
    R.([nm{k} '_sim'])  = Ts(1:n, k);
    R.([nm{k} '_res'])  = m - Ts(1:n, k);
end
fn = {'f_gearbox', 'f_gen_bearing', 'f_hydraulic', 'f_transformer'};
for j = 1:4, R.(fn{j}) = F(1:n, j); end
for c = {'status_type_id', 'in_event'}            % evaluation only
    if ismember(c{1}, D.Properties.VariableNames)
        x = nan(n, 1); x(hit(1:n)) = D.(c{1})(loc(hit(1:n))); R.(c{1}) = x;
    end
end

%% ---- accuracy (normal operation, outside fault windows)
ok = true(n, 1);
if ismember('status_type_id', R.Properties.VariableNames), ok = ok & R.status_type_id == 0; end
if ismember('in_event', R.Properties.VariableNames) && contains(csvFile, '_fault')
    ok = ok & R.in_event == 0;          % leave out the fault window itself
end
fprintf('\nTwin vs measured (%s, %s load, normal operation)\n', csvFile, o.Load);
[r2, rm] = stats(R.P_meas(ok), R.P_sim(ok));
fprintf('  %-13s R2=%6.3f  nRMSE=%5.2f %%\n', 'Power', r2, 100*rm);
for k = 1:5
    [r2, rm] = stats(R.([nm{k} '_meas'])(ok), R.([nm{k} '_sim'])(ok));
    fprintf('  %-13s R2=%6.3f  RMSE =%5.2f C\n', nm{k}, r2, rm);
end

if ~isempty(o.SaveCSV), writetable(R, o.SaveCSV); end
if o.Plot, plot_twin(R, nm); end
end

%% ===================================================== local functions
function A = as_rows(A)
% To Workspace saves a 5x1 column signal as a 5x1xN array.
% Turn every output into N rows x k columns.
A = squeeze(A);
if isvector(A), A = A(:); return; end
if size(A, 1) < size(A, 2), A = A.'; end
end

function y = fill_grid(x, hit, loc)
y = nan(numel(hit), 1);
y(hit) = x(loc(hit));
y = fillmissing(y, 'linear', 'EndValues', 'nearest');
end

function F = fault_profile(td, faults)
F = zeros(numel(td), 4);
col = struct('gearbox', 1, 'gen_bearing', 2, 'hydraulic', 3, 'transformer', 4);
for i = 1:numel(faults)
    c = col.(faults(i).type);
    if faults(i).ramp_days <= 0
        prof = double(td >= faults(i).start_day);
    else
        prof = min(max((td - faults(i).start_day) / faults(i).ramp_days, 0), 1);
    end
    F(:, c) = max(F(:, c), faults(i).severity * prof);
end
end

function [r2, rmse] = stats(y, yh)
m = ~isnan(y) & ~isnan(yh);
r2   = 1 - sum((y(m) - yh(m)).^2) / sum((y(m) - mean(y(m))).^2);
rmse = sqrt(mean((y(m) - yh(m)).^2));
end

function plot_twin(R, nm)
figure('Name', 'Physics twin vs SCADA', 'Color', 'w');
subplot(3, 1, 1);
plot(R.time, R.P_meas, '.', 'MarkerSize', 3); hold on;
plot(R.time, R.P_sim, 'LineWidth', 1); grid on;
ylabel('Power (pu)'); legend('Measured', 'Physics twin'); title('Power');
subplot(3, 1, 2);
plot(R.time, R.([nm{2} '_meas']), '.', 'MarkerSize', 3); hold on;
plot(R.time, R.([nm{2} '_sim']), 'LineWidth', 1);
plot(R.time, R.([nm{3} '_meas']), '.', 'MarkerSize', 3);
plot(R.time, R.([nm{3} '_sim']), 'LineWidth', 1); grid on;
ylabel('\circC'); title('Gearbox HSS bearing and generator bearing');
legend('Gearbox meas', 'Gearbox twin', 'Gen bearing meas', 'Gen bearing twin');
subplot(3, 1, 3);
plot(R.time, R.([nm{5} '_meas']), '.', 'MarkerSize', 3); hold on;
plot(R.time, R.([nm{5} '_sim']), 'LineWidth', 1); grid on;
ylabel('\circC'); title('HV transformer'); legend('Measured', 'Physics twin');
end
