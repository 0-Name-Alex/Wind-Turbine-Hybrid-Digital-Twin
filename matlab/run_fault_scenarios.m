function T = run_fault_scenarios(csvFile, paramsFile)
%RUN_FAULT_SCENARIOS  Part 2: break the twin on purpose and measure the effect.
%   T = run_fault_scenarios;    % defaults: turbine 10 healthy data
%
%   Runs 30 days of real weather through the twin. Each fault starts on
%   day 10 and is Mild / Moderate / Severe, either sudden or growing over
%   1 or 3 days (4 x 3 x 3 = 36 runs + 1 healthy baseline).
%   For each run it reports, over the last 5 days:
%     dT_*      : mean temperature rise vs the healthy run (degC)
%     energy_%  : energy lost vs the healthy run (from fault start)
%     detectable: the shift is larger than the DAILY-averaged model error on
%                 healthy data (99th percentile), for any temperature, or the
%                 power shortfall exceeds the healthy daily power band
%   Results: matlab_results/scenario_summary.csv + one CSV per run.

if nargin < 1, csvFile = fullfile('matlab_data', 'A10_ev03_healthy.csv'); end
if nargin < 2, paramsFile = 'twin_params_A10.mat'; end
outDir = 'matlab_results';
if ~exist(outDir, 'dir'), mkdir(outDir); end

S = load(paramsFile); p = S.p;
opt = {'Days', 30, 'Plot', false};
base = run_twin(csvFile, paramsFile, [], opt{:});
writetable(base, fullfile(outDir, 'scenario_healthy.csv'));

types = {'gearbox', 'gen_bearing', 'hydraulic', 'transformer'};
sev   = [1/3, 2/3, 1];   sevName  = {'Mild', 'Moderate', 'Severe'};
ramp  = [0, 1, 3];       rampName = {'sudden', 'grow1d', 'grow3d'};
last  = base.time >= base.time(end) - days(5);
fault_on = base.time >= base.time(1) + days(10);
nm = p.names;

rows = {};
for i = 1:numel(types)
    for s = 1:3
        for r = 1:3
            f = struct('type', types{i}, 'severity', sev(s), 'start_day', 10, 'ramp_days', ramp(r));
            R = run_twin(csvFile, paramsFile, f, opt{:});
            dT = zeros(1, 5); det = false;
            for k = 1:5
                dT(k) = mean(R.([nm{k} '_sim'])(last) - base.([nm{k} '_sim'])(last), 'omitnan');
                det = det || dT(k) > p.res_day_p99(k);
            end
            e0 = sum(max(base.P_sim(fault_on), 0));
            e1 = sum(max(R.P_sim(fault_on), 0));
            loss = 100 * (e0 - e1) / e0;
            dP  = mean(base.P_sim(last) - R.P_sim(last));
            det = det || dP > abs(p.P_day_p1);           % power shortfall beyond healthy band
            rows(end + 1, :) = [types(i), sevName(s), rampName(r), num2cell([dT, loss]), {det}]; %#ok<AGROW>
            writetable(R, fullfile(outDir, sprintf('scenario_%s_%s_%s.csv', types{i}, sevName{s}, rampName{r})));
        end
    end
end
T = cell2table(rows, 'VariableNames', [{'fault', 'severity', 'onset'}, ...
    strcat('dT_', erase(nm, 'T_')), {'energy_loss_pct', 'detectable'}]);
writetable(T, fullfile(outDir, 'scenario_summary.csv'));
disp(T(strcmp(T.onset, 'sudden'), :));
fprintf('Healthy daily error band (99th pct, degC): %s   power: %.3f pu\n', sprintf('%5.1f', p.res_day_p99), abs(p.P_day_p1));
fprintf('Saved %s\n', fullfile(outDir, 'scenario_summary.csv'));

%% example plot: severe gearbox fault growing over 3 days
R = run_twin(csvFile, paramsFile, struct('type', 'gearbox', 'severity', 1, ...
             'start_day', 10, 'ramp_days', 3), opt{:});
figure('Name', 'Injected gearbox fault', 'Color', 'w');
plot(base.time, base.([nm{2} '_sim']), 'LineWidth', 1); hold on;
plot(R.time, R.([nm{2} '_sim']), 'LineWidth', 1);
plot(R.time, base.([nm{2} '_sim']) + p.res_day_p99(2), 'k--');
grid on; ylabel('Gearbox HSS bearing (\circC)');
legend('Healthy twin', 'Severe gearbox fault (grows over 3 days)', 'Healthy daily error band (p99)');
title('Fault injection in the physics twin');
end
