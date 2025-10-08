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

r = rv(:, 1)
v = rv(:, 2)


