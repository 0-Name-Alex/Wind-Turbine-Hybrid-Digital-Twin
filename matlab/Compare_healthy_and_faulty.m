csv = 'matlab_data/A10_ev03_healthy.csv';  par = 'twin_params_A10.mat';
H = run_twin(csv, par, [], 'Days', 30, 'Plot', false);          % healthy
f = struct('type','gearbox','severity',1,'start_day',10,'ramp_days',3);
F = run_twin(csv, par, f,  'Days', 30, 'Plot', false);          % with fault
figure; plot(H.time, H.T_gb_hss_sim); hold on; plot(F.time, F.T_gb_hss_sim);
legend('Healthy twin','Severe gearbox fault'); ylabel('Gearbox bearing (°C)'); grid on