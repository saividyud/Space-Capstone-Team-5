clear; close all; clc;

%% USER INPUTS (chemical)
g0 = 9.80665; % m/s^2
DeltaV_total = 15000; % total Delta-V required [m/s]
payload_mass = 260; % final useful payload mass after all burns [kg]

% Chemical (lower stage) inputs
Isp_chem = 300; % s 
struct_frac_stage1 = 0.06; % structural fraction of chemical stage initial mass
dV_stage1 = 6000; % m/s 
DeltaV_EP = DeltaV_total - dV_stage1; 

%% EP inputs (for hybrid)
useEP = true; 
Isp_ep = 4000; % s 
eta_thr = 0.6; % thruster electrical-to-kinetic efficiency (0-1)
F_ep = 0.24; % controls burn time
SP_array = 50; % W/kg, solar array specific power (W per kg)
k_ppu = 0.005; % kg/W (mass per watt for PPU) -> 5 kg/kW = 0.005 kg/W
m_thruster_guess = 20; % kg, initial thruster mass estimate
bus_struct_mass = 100; % kg, mass of bus/structure remaining after jettison (not payload)

%% ------------------ CHEMICAL LOWER-STAGE (single-stage-like calc for lower) ------------------
% This block computes the lower stage initial mass if you planned to support
% the lower-stage as a single-stage to accelerate the "bus" (payload+bus_struct).
% For clarity we show the "top-of-chemical-burn" mass (mass EP must accelerate).
% Use simple mass-ratio for chemical lower stage (no upper chemical stage here).

% Compute chemical mass ratio required to provide dV_stage1 for the mass that EP will later accelerate.
% We must do this top-down: we don't yet know EP hardware & prop so iterate later.
% For simplicity: compute chemical stage assuming it accelerates (payload + bus_struct + EP_hw_guess + EP_prop_guess)
% but because EP hardware depends on chemical outcome, here we'll do a pragmatic approach:
fprintf('\n--- Chemical lower-stage summary ---\n');
fprintf('Lower-stage dV = %.0f m/s (Isp_chem = %.0f s)\n', dV_stage1, Isp_chem);

% We'll compute the chemical stage assuming no EP hardware first to get a baseline top-of-burn mass:
R_chem = exp(dV_stage1 / (Isp_chem * g0));
% mf_chem_baseline = payload_mass + bus_struct_mass; % mass after lower stage (mass EP must accelerate)
m0_chem_baseline = R_chem * (payload_mass + bus_struct_mass);
prop_chem_baseline = m0_chem_baseline - (payload_mass + bus_struct_mass);
fprintf('Baseline top-of-chemical-burn mass (no EP hw): m0_chem = %.1f kg, chem prop = %.1f kg\n', m0_chem_baseline, prop_chem_baseline);

%% ------------------ ELECTRIC PROPULSION LEG (iterative sizing) ------------------
if useEP
    if DeltaV_EP <= 0
        error('DeltaV_EP must be positive for EP leg. Adjust dV_stage1.');
    end

    % Helper constants
    ve = Isp_ep * g0;                 % exhaust velocity (m/s)
    R_ep = exp(DeltaV_EP / (Isp_ep * g0));
    prop_frac_ep = 1 - 1 / R_ep;      % fraction of EP-leg initial mass that is propellant (Mp/m0_ep)

    % Iteration variables
    % We model the EP-leg initial mass m0_ep = payload_mass + bus_struct_mass + m_hw + Mp
    % and mf_ep = payload_mass + bus_struct_mass + m_hw (hardware remains)
    % Mp = prop_frac_ep * m0_ep -> Mp = factor * mf -> Mp = (prop_frac_ep/(1-prop_frac_ep)) * mf
    thruster_mass = m_thruster_guess;
    % initial guess for hardware masses (array, ppu) (will be updated)
    m_array = 0;
    m_ppu = 0;
    m_hw = thruster_mass + m_ppu + m_array;
    mf = payload_mass + bus_struct_mass + m_hw;
    tol = 1e-3;
    max_iter = 200;
    converged = false;
    for iter = 1:max_iter
        % compute Mp from mf
        Mp = (prop_frac_ep / (1 - prop_frac_ep)) * mf;   % derived algebra (Mp = factor * mf)
        % mass flow and burn time
        mdot = F_ep / ve;           % kg/s
        if mdot <= 0
            error('Invalid mdot (<=0). Check F_ep and Isp_ep.');
        end
        t_burn = Mp / mdot;         % s

        % required electrical input power (W)
        P_in = 0.5 * F_ep * ve / eta_thr;

        % hardware masses
        m_array_new = P_in / SP_array;            % kg
        m_ppu_new = P_in * k_ppu;                 % kg
        thruster_mass = max(thruster_mass, m_thruster_guess); % keep thruster at least guess
        m_hw_new = thruster_mass + m_ppu_new + m_array_new;

        % update mf and check convergence
        mf_new = payload_mass + bus_struct_mass + m_hw_new;
        if abs(mf_new - mf) < tol
            % converged
            mf = mf_new;
            m_array = m_array_new;
            m_ppu = m_ppu_new;
            m_hw = m_hw_new;
            converged = true;
            break;
        end
        % update and continue
        mf = mf_new;
    end

    if ~converged
        warning('EP sizing did not converge within %d iterations; using last estimate.', max_iter);
    end

    % final numbers
    Mp_final = (prop_frac_ep / (1 - prop_frac_ep)) * mf;
    m0_ep = mf + Mp_final;
    prop_fraction_ep = Mp_final / m0_ep;

    fprintf('\n--- Electric propulsion leg (after jettison) ---\n');
    fprintf('DeltaV_EP = %.0f m/s, Isp_ep = %.0f s\n', DeltaV_EP, Isp_ep);
    fprintf('EP initial mass (m0_ep) = %.1f kg\n', m0_ep);
    fprintf('  - Payload = %.1f kg\n', payload_mass);
    fprintf('  - Bus structure = %.1f kg\n', bus_struct_mass);
    fprintf('  - EP thruster mass (fixed) = %.1f kg\n', thruster_mass);
    fprintf('  - EP PPU mass = %.1f kg\n', m_ppu);
    fprintf('  - Array mass = %.1f kg\n', m_array);
    fprintf('Propellant mass (EP) = %.1f kg (prop fraction = %.4f)\n', Mp_final, prop_fraction_ep);
    % burn time in days
    fprintf('Chosen EP thrust = %.3f N -> burn time = %.2f days (%.1f seconds)\n', F_ep, Mp_final/mdot/86400, Mp_final/mdot);
    fprintf('Electrical input power required = %.1f kW\n', P_in/1000);
else
    fprintf('\nEP leg not requested (useEP = false).\n');
end

%% ------------------ Combine chemical + EP (show total initial mass to launch) ------------------
% If chemical baseline computed earlier is to accelerate the bus (payload+bus_struct+ep_hw+ep_prop),
% the chemical stage initial mass must be recalculated using m0_chem = R_chem*(m_after_chem)
% Where m_after_chem is the mass that EP must accelerate = m0_ep (if EP exists)
if useEP
    % recompute chemical lower-stage masses accounting for EP initial mass being what lower stage must deliver
    mf_chem = m0_ep;  % mass after lower-stage burn; lower stage must accelerate the full EP-initial mass
    m0_chem = R_chem * mf_chem;
    prop_chem = m0_chem - mf_chem;
    fprintf('\n--- Combined system ---\n');
    fprintf('Lower-stage initial mass (m0_chem) to deliver EP-equipped bus = %.1f kg\n', m0_chem);
    fprintf('Lower-stage propellant = %.1f kg\n', prop_chem);
    fprintf('--------------------------------------\n');
fprintf('TOTAL SPACECRAFT LAUNCH MASS\n');
fprintf('--------------------------------------\n');
fprintf('Launch mass (full chemical stage + upper stack) = %.1f kg\n', m0_chem);
fprintf('  Chemical wet mass (m0_chem)     = %.1f kg\n', m0_chem);
fprintf('  Chemical burnout mass (mf_chem) = %.1f kg\n', mf_chem);
fprintf('  Upper stack mass (EP initial mass) = %.1f kg\n', m0_ep);
fprintf('--------------------------------------\n');

else
    fprintf('\nNo EP leg included; chemical baseline only was shown earlier.\n');
end

fprintf('\n(End of hybrid_chem_ep.m)\n');
