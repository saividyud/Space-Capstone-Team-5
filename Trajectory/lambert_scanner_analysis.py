# Imports
import Functions as Funcs
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from tqdm import tqdm
import csv

# Defining Spacecraft parking orbit elements
sc_elements = {
    'a': 3 * u.au,
    'e': 0,
    'i': 45 * u.deg,
    'Omega': 0 * u.deg,
    'omega': 0 * u.deg,
    'nu': 0 * u.deg
}

sim_params = {
    't_burn': -1200 * u.day,
    'nu_burn': 180 * u.deg, # S/C at aphelion/perihelion etc
    'max_flight_time': 1000 * u.day,
    'max_distance': 20 * u.au,
    'a_electric': 0 * u.mm / u.s**2,  # mm/s^2 (Low thrust)
    
    # This vector is what you would iterate on to minimize miss distance
    'dv_chemical_vector': np.array([-1, 0, 5.5]) * u.km / u.s # Defined in the spacecraft orbital frame
}

nu_burns = np.linspace(0, 360, 100, endpoint=False) * u.deg

# Defining vector of time of burns with respect to ISO perihelion
t_burns = np.linspace(-(3*u.yr).to(u.day).value, -100, 100) * u.day

minimum_rel_velocities = []
time_minimum_rel_velocities = []
delta_v_at_minimum_rel_velocities = []

with tqdm(total=len(t_burns)*len(nu_burns), desc="Processing t_burns") as pbar:
    for t_burn in t_burns:
        constant_t_burn_rel_velocities = []
        constant_t_burn_times = []
        constant_t_burn_delta_vs = []

        for nu_burn in nu_burns:
            sim_params = {
                't_burn': t_burn,
                'nu_burn': nu_burn, # S/C at aphelion/perihelion etc
                'max_flight_time': 1000 * u.day,
                'max_distance': 20 * u.au,
                'a_electric': 0 * u.mm / u.s**2,  # mm/s^2 (Low thrust)
                
                # This vector is what you would iterate on to minimize miss distance
                'dv_chemical_vector': np.array([-1, 0, 5.5]) * u.km / u.s # Defined in the spacecraft orbital frame
            }

            mission = Funcs.InterceptorMission(iso_elements=Funcs.characteristic_heliocentric_elements,
                                            sc_elements=sc_elements,
                                            sim_params=sim_params)
            anim1, anim2 = mission.solve_intercept_lambert_scanner(animate=False, steps=10, verbose=False)

            delta_v_mags = np.linalg.norm(mission.dv_history_array, axis=1).value
            relative_velocity_mags = mission.rel_v_of_min_distance_history_array.value

            below_v_threshold_index = np.where(delta_v_mags <= 15)

            if len(below_v_threshold_index[0]) == 0:
                constant_t_burn_rel_velocities.append(np.nan)
                constant_t_burn_times.append(np.nan)
                constant_t_burn_delta_vs.append(np.nan)

                pbar.update(1)
                continue

            min_rel_velocity_index = np.argmin(relative_velocity_mags[below_v_threshold_index])
            min_rel_velocity = relative_velocity_mags[below_v_threshold_index][min_rel_velocity_index]
            time_min_rel_velocity = mission.scanner_times_array.to(u.day)[below_v_threshold_index][min_rel_velocity_index]
            delta_v_min_rel_velocity = delta_v_mags[below_v_threshold_index][min_rel_velocity_index]

            constant_t_burn_rel_velocities.append(min_rel_velocity)
            constant_t_burn_times.append(time_min_rel_velocity)
            constant_t_burn_delta_vs.append(delta_v_min_rel_velocity)

            pbar.update(1)

        minimum_rel_velocities.append(constant_t_burn_rel_velocities)
        time_minimum_rel_velocities.append(constant_t_burn_times)
        delta_v_at_minimum_rel_velocities.append(constant_t_burn_delta_vs)

with open('./Trajectory/lambert_scanner_results_full.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['t_burn (days)', 'nu_burn (degrees)', 'Minimum Relative Velocity (km/s)', 'Time at Minimum Relative Velocity (days)', 'Delta-V at Minimum Relative Velocity (km/s)'])
    
    for t_burn_index, t_burn in enumerate(t_burns):
        for nu_burn_index, nu_burn in enumerate(nu_burns):
            min_rel_vel = minimum_rel_velocities[t_burn_index][nu_burn_index]
            time_min_rel_vel = time_minimum_rel_velocities[t_burn_index][nu_burn_index]
            delta_v_min_rel_vel = delta_v_at_minimum_rel_velocities[t_burn_index][nu_burn_index]
            writer.writerow([t_burn.to(u.day).value, nu_burn.to(u.deg).value, min_rel_vel, time_min_rel_vel.to(u.day).value if isinstance(time_min_rel_vel, u.Quantity) else time_min_rel_vel, delta_v_min_rel_vel])