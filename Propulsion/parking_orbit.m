close all
clear
clc

% Patched conics delta v: LEO to heliocentric circular orbit

% Constants
mu_S = 1.3271244e20; % [m^3/s] Heliocentric
mu_E = 3.986e14; % [m^3/s^2] Geocentric
R_E = 6378.1363e3; 
au2m = 1.496e11; % [m] Astronomical Unit

leo_alt = 400e3;

% Radii
r_LEO = R_E + leo_alt;
r1 = 1 * au2m; % Earth's heliocentric distance (m)
target_r = 3;
r2 = target_r * au2m; % Target heliocentric distance

% Transfer orbit semi-major axis
a_trans = 0.5 * (r1 + r2);

% Speeds
v_E = sqrt(mu_S / r1); 
v_LEO = sqrt(mu_E / r_LEO);
v_esc = sqrt(2*mu_E / r_LEO);

v_t1 = sqrt(mu_S * (2/r1 - 1/a_trans)); % transfer speed at 1 AU (m/s)
v_t2 = sqrt(mu_S * (2/r2 - 1/a_trans)); % transfer speed at r2 (m/s)

v_circ2 = sqrt(mu_S / r2); % circular speed at r2 (m/s)

% Hyperbolic excess at Earth
v_inf = abs(v_t1 - v_E);
v_p = sqrt(v_inf^2 + v_esc^2);

dv_escape = v_p - v_LEO;
dv_insert = v_circ2 - v_t2;

dv_total = abs(dv_escape) + abs(dv_insert);

% Transfer time
% Calculate transfer time using the Hohmann transfer equation
transfer_time = pi * sqrt((a_trans^3) / mu_S);

% Print results
fprintf('LEO altitude: %.0f km\n', leo_alt/1e3);
fprintf('Target radius: %.3f AU\n', target_r);

fprintf('\nv_inf: %.4f km/s\n', v_inf/1e3);

fprintf('Δv_escape (LEO burn): %.4f km/s\n', abs(dv_escape)/1e3);
fprintf('Δv_insert (heliocentric circularization): %.4f km/s\n', abs(dv_insert)/1e3);
fprintf('Total Δv: %.4f km/s\n', dv_total/1e3);

fprintf('Transfer time: %.4f days\n', transfer_time/86400);
