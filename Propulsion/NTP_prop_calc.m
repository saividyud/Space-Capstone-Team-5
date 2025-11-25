close all
clear
clc

%% USER INPUTS (NTP)
g0 = 9.80665; % m/s^2
DeltaV = 15000; % total mission Delta-V [m/s]
payload_mass = 260; % kg (science payload, instruments)

% Nuclear Thermal Propulsion parameters
Isp_ntp = 850; % s  (typical 800–1000)
struct_frac_ntp = 0.12; % fraction of initial mass that is dry structure 
                        % (reactor + tankage + nozzle + shielding)

%% ===================== ROCKET EQUATION ========================
% Ideal mass ratio (no structure)
R_ideal = exp(DeltaV / (Isp_ntp * g0));
prop_frac_ideal = 1 - 1/R_ideal;

% We solve:
%   m0 = initial mass
%   m_struct = struct_frac_ntp * m0
%   mf = payload + m_struct
%   m0 = R_ideal * mf = R_ideal * (payload + struct_frac_ntp*m0)
%
% Rearranged:
%   m0 * (1 - R_ideal * struct_frac_ntp) = R_ideal * payload_mass

denom = 1 - R_ideal * struct_frac_ntp;

fprintf('\n================ NTP SINGLE-STAGE MASS ESTIMATION ================\n');
fprintf('Delta-V Required: %.0f m/s\n', DeltaV);
fprintf('Payload Mass     : %.0f kg\n', payload_mass);
fprintf('Isp (NTP)        : %.0f s\n', Isp_ntp);
fprintf('Structure Fraction: %.3f\n', struct_frac_ntp);

if denom <= 0
    fprintf('\nSINGLE-STAGE NTP INFEASIBLE:\n');
    fprintf('Required mass ratio is too large for the chosen structure fraction.\n');
    fprintf('Try increasing Isp or reducing Delta-V or structural fraction.\n\n');
    
    m0_ntp = NaN; 
    prop_ntp = NaN;
else
    % Solve for initial mass
    m0_ntp = R_ideal * payload_mass / denom;

    m_struct = struct_frac_ntp * m0_ntp;
    mf = payload_mass + m_struct;

    prop_ntp = m0_ntp - mf;
    prop_frac_ntp = prop_ntp / m0_ntp;

    %% ===================== PRINT RESULTS ==========================
    fprintf('\n============= RESULTS: NTP SINGLE STAGE =============\n');
    fprintf('Initial Mass (wet)      : %.1f kg\n', m0_ntp);
    fprintf('Final Mass (dry+payload): %.1f kg\n', mf);
    fprintf('  Payload Mass          : %.1f kg\n', payload_mass);
    fprintf('  Structure Mass (dry)  : %.1f kg\n', m_struct);
    fprintf('Propellant Mass (H2)    : %.1f kg\n', prop_ntp);
    fprintf('Propellant Fraction     : %.4f\n', prop_frac_ntp);
    fprintf('Ideal prop frac (no structure): %.4f\n', prop_frac_ideal);
end

fprintf('\n======================== END ========================\n');
