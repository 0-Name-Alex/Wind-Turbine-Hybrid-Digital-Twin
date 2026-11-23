function D = load_scada(f)
%LOAD_SCADA  Read one CSV made by export_for_matlab.py.
%   Temperatures above 200 degC (sensor error code 205) become NaN.
%   Inputs (wind, temperatures of the air, power, rotor speed) are gap-filled.
opts = detectImportOptions(f);
opts = setvartype(opts, 'time', 'datetime');
opts = setvaropts(opts, 'time', 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
opts = setvartype(opts, 'train_test', 'string');
D = readtable(f, opts);

v = D.Properties.VariableNames;
for i = 1:numel(v)
    if startsWith(v{i}, 'T_')
        x = D.(v{i});
        x(x > 200) = NaN;
        D.(v{i}) = x;
    end
end
for nm = {'wind', 'T_amb', 'T_nacelle', 'P', 'rotor_rpm', 'pitch'}
    D.(nm{1}) = fillmissing(D.(nm{1}), 'linear', 'EndValues', 'nearest');
end
D.w = D.rotor_rpm / 14.9;          % per-unit rotor speed
end
