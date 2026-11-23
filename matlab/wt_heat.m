function [P_grid, Teq] = wt_heat(P, w, Tnac, Tamb, f, C, kf)
%WT_HEAT  Losses -> equilibrium temperature of each component (with faults).
%   Components (rows of C): 1 gearbox oil, 2 gearbox HSS bearing,
%                           3 generator bearing (DE), 4 stator winding,
%                           5 HV transformer
%   Columns of C: [P  P^2  w  T_nacelle  T_ambient  1]
%       P   -> load-dependent losses (gear mesh)
%       P^2 -> copper losses (current squared)
%       w   -> speed-dependent losses (bearing friction, oil churning, iron)
%   Each coefficient = thermal resistance x loss coefficient, in degC.
%   f  = fault severity [gearbox gen_bearing hydraulic transformer], 0..1
%   kf = fault strengths [k_gb_heat k_gb_eta k_brg k_hyd_deg k_trafo]
%#codegen
Pp = max(P, 0);
X  = [Pp; Pp^2; w; Tnac; Tamb; 1];
Cf = C;
Cf(1:2, 1:3) = C(1:2, 1:3)*(1 + kf(1)*f(1));   % gearbox: more mesh/bearing loss
Cf(3, 3)     = C(3, 3)*(1 + kf(3)*f(2));       % generator bearing: more friction
Cf(5, 2)     = C(5, 2)*(1 + kf(5)*f(4));       % transformer: more winding loss
Teq    = Cf*X;
P_grid = P*(1 - kf(2)*f(1));                   % gearbox: efficiency drop
end
