close all
clear
clc

%% USER INPUTS
g0 = 9.80665; % [m/s^2]
DeltaV_total = 15000; % total Delta-V required [m/s]

payload_mass = 260; % final useful payload mass (instruments, science payload) after all burns [kg]

% SINGLE-STAGE inputs
Isp1 = 400; % [s]
struct_frac_stage1 = 0.06; % structure fraction of stage1 initial mass (e.g. 0.05-0.10)

% OPTIONAL TWO-STAGE inputs (set Isp2>0 to run two-stage)
Isp2 = 400; % [s] (upper-stage Isp)
struct_frac_stage2 = 0.06; % structure fraction for upper stage (fraction of that stage's initial mass)
dV_stage1 = 6000; % [m/s] for lower-stage burn
dV_stage2 = DeltaV_total - dV_stage1; % remaining for upper-stage burn

%% SINGLE-STAGE CALCULATION (no jettison)
% Mass ratio required (ideal, no structure)
R_single = exp(DeltaV_total / (Isp1 * g0));   % m0/mf (mf = final mass after burn)
prop_frac_ideal = 1 - 1 / R_single;

% Now consider structure for the single stage
%   m0 = initial stage mass (unknown),
%   mstruct = struct_frac_stage1 * m0,
%   payload_mass is final useful mass (payload),
%   mf = payload_mass + mstruct
% Using m0 = R_single * mf -> m0 = R_single * (payload_mass + struct_frac_stage1*m0)
% Solve for m0: m0 * (1 - R_single * struct_frac_stage1) = R_single * payload_mass

denom = 1 - R_single * struct_frac_stage1;
if denom <= 0
    fprintf('Single-stage infeasible: structural fraction too large for required mass ratio (denom<=0).\n');
    m0_single = NaN;
    prop_mass_single = NaN;
    prop_frac_single = NaN;
else
    m0_single = R_single * payload_mass / denom;
    mstruct_single = struct_frac_stage1 * m0_single;
    mf_single = payload_mass + mstruct_single;
    prop_mass_single = m0_single - mf_single;
    prop_frac_single = prop_mass_single / m0_single;
end

%% OPTIONAL TWO-STAGE CALCULATION
if exist('Isp2','var') && ~isempty(Isp2) && Isp2 > 0
    % Compute for upper stage first (top-down)
    R2 = exp(dV_stage2 / (Isp2 * g0));
    denom2 = 1 - R2 * struct_frac_stage2;
    if denom2 <= 0
        fprintf('Two-stage infeasible: upper stage structural fraction too large (denom2<=0).\n');
        m0_total = NaN; total_prop = NaN; prop_frac_total = NaN;
    else
        % Upper stage initial mass (includes its structure and prop)
        m0_upper = R2 * payload_mass / denom2;
        mstruct_upper = struct_frac_stage2 * m0_upper;
        mf_upper = payload_mass + mstruct_upper;  % mass left after upper burn (payload + upper structure)
        prop_upper = m0_upper - mf_upper;

        % Lower stage treats m0_upper as the "payload" it must accelerate
        R1 = exp(dV_stage1 / (Isp1 * g0));
        denom1 = 1 - R1 * struct_frac_stage1;
        if denom1 <= 0
            fprintf('Two-stage infeasible: lower stage structural fraction too large (denom1<=0).\n');
            m0_total = NaN; total_prop = NaN; prop_frac_total = NaN;
        else
            m0_total = R1 * m0_upper / denom1;   % initial mass of whole vehicle
            mstruct_lower = struct_frac_stage1 * m0_total;
            mf_after_lower = m0_upper + mstruct_lower; % mass after lower burn (upper init mass + lower struct)
            prop_lower = m0_total - mf_after_lower;

            % totals
            total_prop = prop_lower + prop_upper;
            prop_frac_total = total_prop / m0_total;
        end
    end
else
    % No two-stage requested
    m0_total = NaN; total_prop = NaN; prop_frac_total = NaN;
end

%% PRINT RESULTS
fprintf('\nRESULTS\n');
fprintf('Total required Delta-V = %.0f m/s\n', DeltaV_total);
fprintf('Payload Mass = %.0f kg\n', payload_mass);

fprintf('\nSingle-stage (Isp = %.0f s): \n', Isp1);
if isnan(m0_single)
    fprintf(' Single-stage infeasible with struct frac = %.2f\n', struct_frac_stage1);
else
    fprintf(' Initial mass m0 = %.1f kg\n', m0_single);
    fprintf(' Structure mass (stage) = %.1f kg\n', mstruct_single);
    fprintf(' Propellant mass = %.1f kg\n', prop_mass_single);
    fprintf(' Propellant fraction of initial mass = %.4f\n', prop_frac_single);
    fprintf(' Ideal prop fraction (no structure) = %.4f\n', prop_frac_ideal);
end

if ~isnan(m0_total)
    fprintf('\nTwo-stage (Isp1 = %.0f s, Isp2 = %.0f s): \n', Isp1, Isp2);
    fprintf(' Lower-stage dV = %.0f m/s, Upper-stage dV = %.0f m/s\n', dV_stage1, dV_stage2);
    fprintf(' Total initial mass m0 = %.1f kg\n', m0_total);
    fprintf(' Lower-stage propellant = %.1f kg\n', prop_lower);
    fprintf(' Upper-stage propellant = %.1f kg\n', prop_upper);
    fprintf(' Total propellant = %.1f kg\n', total_prop);
    fprintf(' Total propellant fraction = %.4f\n', prop_frac_total);
else
    if exist('Isp2','var') && ~isempty(Isp2) && Isp2 > 0
        fprintf('\nTwo-stage calculation was infeasible with the chosen structural fractions or dV split.\n');
    end
end


fprintf('\nTOTAL INITIAL MASS SUMMARY\n');

% Single-stage
if ~isnan(m0_single)
    % m0_single already computed
    % prop_mass_single already computed
    fprintf('Single-stage initial mass (wet): %.1f kg\n', m0_single);
    fprintf('Single-stage propellant mass: %.1f kg\n', prop_mass_single);
else
    fprintf('Single-stage propellant mass: infeasible\n');
end

% Two-stage
if ~isnan(m0_total)
    % m0_total already computed
    % total_prop already computed
    fprintf('\nTwo-stage initial mass (wet): %.1f kg\n', m0_total);
    fprintf('Two-stage total propellant mass: %.1f kg\n', total_prop);
else
    fprintf('\nTwo-stage propellant mass: infeasible\n');
end