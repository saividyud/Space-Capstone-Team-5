# Imports
import Functions as Funcs
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

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
    't_burn': 40 * u.day,   # Burn executed at 700 days before perihelion
    'nu_burn': 0 * u.deg, # S/C at aphelion/perihelion etc
    'max_flight_time': 1000 * u.day,
    'max_distance': 20 * u.au,
    'a_electric': 0 * u.mm / u.s**2,  # mm/s^2 (Low thrust)
    
    # This vector is what you would iterate on to minimize miss distance
    'dv_chemical_vector': np.array([-1, 0, 5.5]) * u.km / u.s # Defined in the spacecraft orbital frame
}

print(sim_params)
mission = Funcs.InterceptorMission(iso_elements=Funcs.characteristic_heliocentric_elements,
                                  sc_elements=sc_elements,
                                  sim_params=sim_params)

# mission.solve_intercept_lambert()
# mission.intercept_trajectory()
# anim = mission.plot_trajectory()

anim1, anim2 = mission.solve_intercept_lambert_scanner(animate=True)

# anim1.save('lambert_scanner_trajectory_example.mp4', writer='ffmpeg', fps=60)
# anim2.save('lambert_scanner_history.mp4', writer='ffmpeg', fps=60)

# fig = plt.figure(figsize=(10, 8))
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
# ax = fig.add_subplot()

# after_perihelion_index = np.where(mission.scanner_times_array.to(u.day).value >= -200)

# delta_v_mags = np.linalg.norm(mission.dv_history_array, axis=1)[after_perihelion_index].value
# relative_velocity_mags = mission.rel_v_of_min_distance_history_array[after_perihelion_index].value

# ax.plot(mission.scanner_times_array.to(u.day)[after_perihelion_index].value, delta_v_mags, label='Initial Burn Delta-V Magnitude')
# ax.plot(mission.scanner_times_array.to(u.day)[after_perihelion_index].value, relative_velocity_mags, label='Relative Velocity at Min Distance')
# # ax.axhline(y=np.min(relative_velocity_mags), color='r', linestyle='--', label='Minimum Relative Velocity')
# ax.axhline(y=15, color='g', linestyle='--', label='15 km/s Threshold')

# ax.set_xlabel('Intercept Time since ISO Perihelion (days)')
# ax.set_ylabel('Velocity (km/s)')
# ax.set_title('Delta-V and Relative Velocity vs Intercept Time')

# ax.grid(True)

# ax.legend(fontsize=12)

# ax2 = fig.add_subplot(2, 1, 2)
# min_distances = mission.min_distance_history_array[after_perihelion_index].to(u.km).value
# ax2.plot(mission.scanner_times_array.to(u.day)[after_perihelion_index].value, min_distances, color='orange', label='Minimum Distance to ISO')
# ax2.set_xlabel('Intercept Time since ISO Perihelion (days)')
# ax2.set_ylabel('Minimum Distance (km)')
# ax2.set_title('Minimum Distance to ISO vs Intercept Time')
# ax2.axhline(y=1e3, color='r', linestyle='--', label='1,000 km Threshold')
# ax2.grid(True)
# ax2.legend(fontsize=12)

# fig = plt.figure()
# ax = fig.add_subplot()

# ax.plot(mission.iso_ts * u.s.to(u.day), np.linalg.norm(mission.characteristic_iso_states[:, :3], axis=1), label='ISO Distance')

# t_span = np.array([sim_params['t_burn'].to(u.day).value, sim_params['max_flight_time'].to(u.day).value]) * u.day
# print(sim_params['max_flight_time'])

# r0 = np.array([-2.29813333,  1.36355843,  1.36355843]) * u.au
# # v0 = np.array([-11.05350419,  -9.31475534,  -9.31475534]) * u.km / u.s
# v0 = np.array([  1.25973059, -12.37875423, -19.17026888]) * u.km / u.s

# sol = mission.propagate_intercept(t_span, np.concatenate((r0.to(u.km).value, v0.to(u.km / u.s).value)), sim_params['a_electric'])

# rs = (sol.y[0:3] * u.km).to(u.au).value.T

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(rs[:, 0], rs[:, 1], rs[:, 2], label='Spacecraft Trajectory')
# ax.set_xlabel('X (au)')
# ax.set_ylabel('Y (au)')
# ax.set_zlabel('Z (au)')
# ax.set_title('Spacecraft Intercept Trajectory')
# ax.legend()

# ax.set_xlim([-5, 5])
# ax.set_ylim([-5, 5])
# ax.set_zlim([-5, 5])

# ax.set_aspect('equal')

plt.show()