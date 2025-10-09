clear
clc
close all

% This script tests Kumar's RV from OE and OE from RV codes and compares it
% to my code in Python to ensure accuracy

%% Test Case 1
a = -1.27234500742808e+01 * 1.496e+8; % [km]
e = 1.201133796102373e+01;
i = 122.7417062847286; % [deg]
Omega = 24.59690955523242; % [deg]
omega = 241.8105360304898; % [deg]
f = 0.08802938773805373 * 180/pi; % [deg]

rv = RVfromOE([a, e, deg2rad(i), deg2rad(Omega), deg2rad(omega), deg2rad(f)]);

rv(:, 1);
rv(:, 2);

%% Test Case 2
r = [-1.18713059e+10; 6.07080662e+09; -1.62676916e+10];
v = [6.33710179; 5.43131958; -3.5558986];

oe = OEfromRV(r, v);

a = oe(1) / 1.496e+8
e = oe(2)
i = rad2deg(oe(3))
Omega = rad2deg(oe(4))
omega = rad2deg(oe(5))
f = rad2deg(oe(6))