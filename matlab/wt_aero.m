function [P, w_pu, beta] = wt_aero(v, beta_off, Kp, R, wmin, Prated, Pstandby)
%WT_AERO  Rotor aerodynamics + turbine control for one 10-min step.
%   Below rated : rotor speed follows the optimum tip-speed ratio (MPPT),
%                 limited between wmin and rated speed.
%   Above rated : pitch angle is increased until power = rated power.
%   beta_off    : pitch misalignment (deg) - used for the hydraulic fault.
%   Outputs     : P (per-unit power), w_pu (per-unit rotor speed), beta (deg)
%#codegen
w_rated = 14.9*2*pi/60;          % rated rotor speed from SCADA (14.9 rpm)
lam_opt = 8.1;                   % optimum tip-speed ratio of the Heier Cp curve
if v < 3.5 || v > 25             % below cut-in / above cut-out
    P = Pstandby; w_pu = 0; beta = 0;
    return
end
w    = min(max(lam_opt*v/R, wmin*w_rated), w_rated);
lam  = w*R/v;
Pa   = Kp*wt_cp(lam, beta_off)*v^3;
beta = beta_off;
P    = Pa;
if Pa > Prated                   % pitch control: bisection for the pitch angle
    lo = beta_off; hi = 40;
    for i = 1:30
        b = (lo + hi)/2;
        if Kp*wt_cp(lam, b)*v^3 > Prated, lo = b; else, hi = b; end
    end
    beta = (lo + hi)/2;
    P    = Prated;
end
w_pu = w/w_rated;
end
