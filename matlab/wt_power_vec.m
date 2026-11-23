function P = wt_power_vec(v, Kp, R, wmin, Prated, Pstandby)
%WT_POWER_VEC  Vectorised version of wt_aero (power only, no fault) for calibration.
w_rated = 14.9*2*pi/60;
w   = min(max(8.1*v/R, wmin*w_rated), w_rated);
lam = w.*R ./ max(v, 0.1);
P   = min(Kp*wt_cp(lam, 0).*v.^3, Prated);
P(v < 3.5 | v > 25) = Pstandby;
end
