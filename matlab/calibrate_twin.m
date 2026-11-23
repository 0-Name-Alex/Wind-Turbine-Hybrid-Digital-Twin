function p = calibrate_twin(asset, dataDir)
%CALIBRATE_TWIN  Tune the physics twin to one turbine's HEALTHY data.
%   p = calibrate_twin(10)              % turbine 10, CSVs in .\matlab_data
%   p = calibrate_twin(10, 'my_folder')
%
%   Uses only the "train" part of that turbine's healthy datasets.
%   Power fit : normal-operation rows only (status 0), robust loss so that
%               curtailed / derated periods do not bend the curve.
%   Thermal   : for each component, a grid search over the time constant tau
%               plus a linear least-squares fit of the loss coefficients.
%   Base MATLAB only (fminsearch, filter, backslash) - no toolboxes needed.
%   Saves twin_params_Axx.mat

if nargin < 1, asset = 10; end
if nargin < 2, dataDir = 'matlab_data'; end

files = dir(fullfile(dataDir, sprintf('A%02d_ev*_healthy.csv', asset)));
assert(~isempty(files), 'No healthy files for asset %d in %s', asset, dataDir);

D = [];
for i = 1:numel(files)
    Di = load_scada(fullfile(files(i).folder, files(i).name));
    D  = [D; Di(Di.train_test == "train", :)]; %#ok<AGROW>
    fprintf('Loaded %s\n', files(i).name);
end
[~, iu] = unique(D.time);               % sorted, duplicates removed
D   = D(iu, :);
seg = cumsum([true; diff(D.time) ~= minutes(10)]);   % continuous blocks

%% ---------------------------------------------------------------- POWER
p.asset    = asset;
p.Pstandby = -0.004;
N  = D.status_type_id == 0 & ~isnan(D.wind) & ~isnan(D.P);
hi = N & D.wind > 16;
p.Prated = 0.975;
if nnz(hi) > 50, p.Prated = median(D.P(hi)); end

v = D.wind(N); y = D.P(N); fs = 0.03;
unpack = @(q) [exp(q(1)), min(max(q(2), 20), 80), min(max(q(3), 0.2), 0.9)];
cost   = @(q) robust_cost(wt_power_vec(v, exp(q(1)), min(max(q(2),20),80), ...
                          min(max(q(3),0.2),0.9), p.Prated, p.Pstandby) - y, fs);
q = fminsearch(cost, [log(1.6e-3), 43, 0.67], ...
               optimset('MaxFunEvals', 3000, 'MaxIter', 3000, 'TolX', 1e-7, 'TolFun', 1e-9));
u = unpack(q);
p.Kp = u(1); p.R = u(2); p.wmin = u(3);
Pm = wt_power_vec(v, p.Kp, p.R, p.wmin, p.Prated, p.Pstandby);
[p.P_r2, p.P_rmse] = fit_stats(y, Pm);
eP = nan(height(D), 1);                                   % daily power shortfall band
eP(N) = y - Pm;
p.P_day_p1 = -q99(-movmean(eP, 144, 'omitnan'));          % 1st percentile (negative)
fprintf('\nPOWER  Kp=%.3g  R=%.1f m  wmin=%.2f pu  Prated=%.3f  ->  R2=%.3f  nRMSE=%.2f%%\n', ...
        p.Kp, p.R, p.wmin, p.Prated, p.P_r2, 100*p.P_rmse);

%% ---------------------------------------------------------------- THERMAL
p.names = {'T_gb_oil', 'T_gb_hss', 'T_genbrg_DE', 'T_stator1', 'T_trafo3'};
%            P  P2  w  Tnac Tamb 1      which loss terms each component has
use = logical([1  0  1   1    1   1;     % gearbox oil
               1  0  1   1    1   1;     % gearbox HSS bearing
               0  1  1   1    1   1;     % generator bearing
               0  1  1   1    1   1;     % stator winding
               0  1  0   1    1   1]);   % transformer
Pp = max(D.P, 0);
X  = [Pp, Pp.^2, D.w, D.T_nacelle, D.T_amb, ones(height(D), 1)];
taus = [0.5 1 2 3 4 6 8 12];                     % hours

p.C = zeros(5, 6); p.tau_h = zeros(5, 1);
p.r2 = zeros(5, 1); p.rmse = zeros(5, 1); p.res_std = zeros(5, 1);
p.res_p99 = zeros(5, 1); p.res_day_p99 = zeros(5, 1);
fprintf('\n%-13s %6s %7s %8s %9s   coefficients [P P2 w Tnac Tamb 1]\n', 'signal', 'tau(h)', 'R2', 'RMSE', 'day-band');
for k = 1:5
    yk = D.(p.names{k});
    best = -Inf;
    for th = taus
        a    = 600/(th*3600);
        B    = filt_seg(X(:, use(k, :)), a, seg);
        base = init_resp(yk, a, seg);
        m    = ~isnan(yk);
        c    = B(m, :) \ (yk(m) - base(m));
        yh   = base + B*c;
        r2   = fit_stats(yk, yh);
        if r2 > best
            best = r2; p.tau_h(k) = th;
            p.C(k, :) = 0; p.C(k, use(k, :)) = c';
            res = yk - yh;
        end
    end
    [p.r2(k), p.rmse(k)] = fit_stats(yk, yk - res);
    p.res_std(k) = std(res, 'omitnan');
    p.res_p99(k) = q99(res);                                  % single 10-min error
    p.res_day_p99(k) = q99(movmean(res, 144, 'omitnan'));     % daily-averaged error
    fprintf('%-13s %6.1f %7.3f %6.2f C %7.1f C   [%s]\n', p.names{k}, p.tau_h(k), p.r2(k), ...
            p.rmse(k), p.res_day_p99(k), sprintf('%6.1f', p.C(k, :)));
end

%% ---------------------------------------------------------------- FAULTS
% Strength of each injected fault at severity = 1 (Severe). These are
% engineering assumptions - state them as such in the report.
p.kfault = [0.5, ...   % gearbox: +50 % gear/bearing losses
            0.02, ...  % gearbox: -2 % efficiency (power lost to heat)
            1.0, ...   % generator bearing: friction doubled
            3.0, ...   % hydraulic pitch: 3 deg blade misalignment
            0.6];      % transformer: +60 % winding losses
fn = sprintf('twin_params_A%02d.mat', asset);
save(fn, 'p');
fprintf('\nSaved %s\n', fn);
end

%% ===================================================== local functions
function J = robust_cost(r, fs)
J = sum(fs^2 * 2 * (sqrt(1 + (r/fs).^2) - 1));     % soft-L1 loss
end

function Y = filt_seg(X, a, seg)
% First-order thermal lag T[k] = (1-a)T[k-1] + a*x[k], restarted at data gaps
Y = zeros(size(X));
e = [find([true; diff(seg) ~= 0]); numel(seg) + 1];
for s = 1:numel(e) - 1
    idx = e(s):e(s + 1) - 1;
    Y(idx, :) = filter(a, [1, -(1 - a)], X(idx, :));
end
end

function r = init_resp(y, a, seg)
% Decay of the measured starting temperature of each continuous block
r = zeros(size(y));
e = [find([true; diff(seg) ~= 0]); numel(seg) + 1];
for s = 1:numel(e) - 1
    idx = e(s):e(s + 1) - 1;
    y0 = y(idx(1));
    if isnan(y0), y0 = mean(y(idx), 'omitnan'); end
    if isnan(y0), y0 = median(y, 'omitnan'); end
    r(idx) = y0 * (1 - a).^(1:numel(idx))';
end
end

function [r2, rmse] = fit_stats(y, yh)
m = ~isnan(y) & ~isnan(yh);
r2   = 1 - sum((y(m) - yh(m)).^2) / sum((y(m) - mean(y(m))).^2);
rmse = sqrt(mean((y(m) - yh(m)).^2));
end

function q = q99(x)
x = sort(x(~isnan(x)));
q = x(max(1, round(0.99*numel(x))));
end
