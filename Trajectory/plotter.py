#%% Imports
import Functions as Funcs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from astropy import units as u
from astropy.time import Time

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['mathtext.fontset'] = 'cm'

#%% Defining Earth orbital elements
earth_elements = {
    'a': 1 * u.au,
    'e': 0.0167,
    'i': 0 * u.deg,
    'Omega': 0 * u.deg,
    'omega': 102.9372 * u.deg,
    'nu': 0 * u.deg
}

# Generating Earth orbit data over one orbital period
t_start = 0
t_end = Funcs.period(earth_elements['a']).to(u.s).value
earth_ts = np.linspace(t_start, t_end, 1000)
earth_states = np.zeros((len(earth_ts), 6))

for i, t in enumerate(earth_ts):
    r_earth, v_earth = Funcs.get_sc_state(t, earth_elements)
    earth_states[i, :3] = r_earth.to(u.au).value
    earth_states[i, 3:] = v_earth.to(u.km / u.s).value

#%% Defining Mars orbital elements
mars_elements = {
    'a': 1.523679 * u.au,
    'e': 0.0934,
    'i': 1.850 * u.deg,
    'Omega': 49.57854 * u.deg,
    'omega': 336.04084 * u.deg,
    'nu': 0 * u.deg
}

# Generating Mars orbit data over one orbital period
t_start = 0
t_end = Funcs.period(mars_elements['a']).to(u.s).value
mars_ts = np.linspace(t_start, t_end, 1000)
mars_states = np.zeros((len(mars_ts), 6))

for i, t in enumerate(mars_ts):
    r_mars, v_mars = Funcs.get_sc_state(t, mars_elements)
    mars_states[i, :3] = r_mars.to(u.au).value
    mars_states[i, 3:] = v_mars.to(u.km / u.s).value

#%% Defining ISO orbital elements
# Defining heliocentric orbital elements for known ISOs
oumuamua_heliocentric_elements = {
    'a': -1.27234500742808 * u.au,  # Semi-major axis
    'e': 1.201133796102373,      # Eccentricity
    'i': 122.7417062847286 * u.deg,  # Inclination
    'Omega': 24.59690955523242 * u.deg,  # Longitude of ascending node
    'omega': 241.8105360304898 * u.deg,  # Argument of perihelion
    'M': 0 * u.deg,  # Mean anomaly
    'epoch': Time('2458006.007321375231', format='jd', scale='tdb')  # Time of perihelion passage
}

borisov_heliocentric_elements = {
    'a': -0.8514922551937886 * u.au,  # Semi-major axis
    'e': 3.356475782676596,      # Eccentricity
    'i': 44.05264247909138 * u.deg,  # Inclination
    'Omega': 308.1477292269942 * u.deg,  # Longitude of ascending node
    'omega': 209.1236864378081 * u.deg,  # Argument of perihelion
    'M': 0 * u.deg,  # Mean anomaly
    'epoch': Time('2458826.052845906059', format='jd', scale='tdb')  # Time of perihelion passage
}

atlas_heliocentric_elements = {
    'a': -0.263915917517816 * u.au,  # Semi-major axis
    'e': 6.139587836355706,      # Eccentricity
    'i': 175.1131015287974 * u.deg,  # Inclination
    'Omega': 322.1568699043938 * u.deg,  # Longitude of ascending node
    'omega': 128.0099421020839 * u.deg,  # Argument of perihelion
    'M': 0 * u.deg,  # Mean anomaly
    'epoch': Time('2460977.981439259343', format='jd', scale='tdb')  # Time of perihelion passage
}

# Calculating characteristic heliocentric elements by averaging known ISOs
characteristic_heliocentric_elements = {
    'a': np.mean([oumuamua_heliocentric_elements['a'].value,
                  borisov_heliocentric_elements['a'].value,
                  atlas_heliocentric_elements['a'].value]) * u.au,
    'e': np.mean([oumuamua_heliocentric_elements['e'],
                  borisov_heliocentric_elements['e'],
                  atlas_heliocentric_elements['e']]),
    'i': np.mean([oumuamua_heliocentric_elements['i'].value,
                  borisov_heliocentric_elements['i'].value,
                  atlas_heliocentric_elements['i'].value]) * u.deg,
    'Omega': np.mean([oumuamua_heliocentric_elements['Omega'].value,
                      borisov_heliocentric_elements['Omega'].value,
                      atlas_heliocentric_elements['Omega'].value]) * u.deg,
    'omega': np.mean([oumuamua_heliocentric_elements['omega'].value,
                      borisov_heliocentric_elements['omega'].value,
                      atlas_heliocentric_elements['omega'].value]) * u.deg,
    'M': 0 * u.deg,  # Mean anomaly
    'epoch': Time('2459000.0', format='jd', scale='tdb')  # Arbitrary epoch
}

#%% Defining Spacecraft parking orbit elements
sc_elements = {
    'a': 0.75 * u.au,
    'e': 0,
    'i': 45 * u.deg,
    'Omega': 0 * u.deg,
    'omega': 0 * u.deg,
    'nu': 0 * u.deg
}

# Generating spacecraft parking orbit data over one orbital period
t_start = 0
t_end = Funcs.period(sc_elements['a']).to(u.s).value
sc_ts = np.linspace(t_start, t_end, 1000)
sc_states = np.zeros((len(sc_ts), 6))

for i, t in enumerate(sc_ts):
    r_sc, v_sc = Funcs.get_sc_state(t, sc_elements)
    sc_states[i, :3] = r_sc.to(u.au).value
    sc_states[i, 3:] = v_sc.to(u.km / u.s).value

#%% Plotting spacecraft intercept trajectory
# Simulation parameters
sim_params = {
    't_detect': -400 * u.day, # Detected 200 days before perihelion
    't_burn': -180 * u.day,   # Burn executed 210 days before perihelion
    'nu_burn': 0 * u.deg, # S/C at aphelion/perihelion etc
    'a_electric': 1e-6 * u.km / u.s**2,  # km/s^2 (Low thrust)
    'max_flight_time': 800 * u.day,
    
    # This vector is what you would iterate on to minimize miss distance
    'dv_chemical_vector': np.array([-3, 0, 10]) * u.km / u.s # Defined in the spacecraft orbital frame
}

# Defining time span for intercept propagation
t_span = np.array([sim_params['t_burn'].value, sim_params['max_flight_time'].value]) * u.day

# Initial state of spacecraft at time of burn
sc_initial_elements = sc_elements.copy()
sc_initial_elements.pop('M', None)
sc_initial_elements['nu'] = sim_params['nu_burn']

# Rotate dv_chemical_vector to inertial frame
sim_params['dv_chemical_vector'] = (Funcs.R_BI(sc_initial_elements).T @ sim_params['dv_chemical_vector'].T).T

r_initial, v_initial = Funcs.OEtoRV(sc_initial_elements)
v_initial += sim_params['dv_chemical_vector']
initial_state = np.concatenate((r_initial.to(u.km).value, v_initial.to(u.km / u.s).value))

# Propagate intercept trajectory
sol = Funcs.propagate_intercept(t_span, initial_state, sim_params['a_electric'])
sc_intercept_ts = sol.t * u.s
sc_intercept_rs = (sol.y[0:3] * u.km).to(u.au).value.T
sc_intercept_vs = (sol.y[3:6] * u.km / u.s).value.T

#%% Defining characteristic ISO trajectory
t_start = sim_params['t_burn'].to(u.s).value  # 200 days before perihelion
t_end = sim_params['max_flight_time'].to(u.s).value     # 200 days after perihelion
iso_ts = np.linspace(t_start, t_end, 1000)
characteristic_iso_states = np.zeros((len(iso_ts), 6))  # Assuming 3 for position and 3 for velocity

for i, t in enumerate(iso_ts):
    r, v = Funcs.get_iso_state(t, characteristic_heliocentric_elements)
    characteristic_iso_states[i, :3] = r.to(u.au).value
    characteristic_iso_states[i, 3:] = v.to(u.km / u.s).value

#%% Calculating attributes of this intercept trajectory
relative_distance = np.linalg.norm(sc_intercept_rs - characteristic_iso_states[:, :3], axis=1) * u.au
relative_velocity = np.linalg.norm(sc_intercept_vs - characteristic_iso_states[:, 3:], axis=1) * u.km / u.s

min_distance_index = np.argmin(relative_distance)

min_distance = relative_distance[min_distance_index]
rel_v_of_min_distance = relative_velocity[min_distance_index]
time_of_min_distance = sc_intercept_ts[min_distance_index]

print(f"Minimum Distance at Intercept: {min_distance:.3f}, {min_distance.to(u.km):.0f}")
print(f"Time of Minimum Distance: {time_of_min_distance.to(u.day):.2f}")
print(f"Relative Velocity at Minimum Distance: {rel_v_of_min_distance:.6f}")

#%% Plotting the trajectories
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(0, 0, 0, color='orange', s=50, label='Sun', marker='*')

ax.plot(earth_states[:, 0], earth_states[:, 1], earth_states[:, 2],
        label='Earth Orbit', color='black', linestyle='--')

ax.plot(mars_states[:, 0], mars_states[:, 1], mars_states[:, 2],
        label='Mars Orbit', color='black', linestyle='--')

ax.plot(characteristic_iso_states[:, 0], characteristic_iso_states[:, 1], characteristic_iso_states[:, 2],
        label='Characteristic ISO Trajectory', color='red')
# ax.scatter(characteristic_iso_states[-1, 0], characteristic_iso_states[-1, 1], characteristic_iso_states[-1, 2],
#            color='red', s=30, label='ISO End', marker='o')
iso_pos, = ax.plot([], [], [], color='red', marker='o', markersize=8, label='ISO Position', linestyle='')

ax.plot(sc_states[:, 0], sc_states[:, 1], sc_states[:, 2],
        label='Spacecraft Parking Orbit', color='blue', linestyle=':')

ax.plot(sc_intercept_rs[:, 0], sc_intercept_rs[:, 1], sc_intercept_rs[:, 2],
        label='Spacecraft Intercept Trajectory', color='blue')
# ax.scatter(sc_intercept_rs[0, 0], sc_intercept_rs[0, 1], sc_intercept_rs[0, 2],
#            color='blue', s=30, label='Spacecraft Boost Point', marker='o')
sc_intercept_pos, = ax.plot([], [], [], color='blue', marker='o', markersize=8, label='S/C Position', linestyle='')

max_dist = np.amax([np.abs(sc_states[:, :3]), np.abs(characteristic_iso_states[:, :3])])

ax.set_xlim([-max_dist, max_dist])
ax.set_ylim([-max_dist, max_dist])
ax.set_zlim([-max_dist, max_dist])

ax.set_aspect('equal')

ax.set_xlabel('X [au]')
ax.set_ylabel('Y [au]')
ax.set_zlabel('Z [au]')
ax.set_title('Trajectory of ISOs and Spacecraft Interception')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)

# Plotting distance over time
ax2 = fig.add_subplot(2, 2, 2)

ax2.plot(sc_intercept_ts.to(u.day).value, relative_distance, color='purple')
ax2.axvline(time_of_min_distance.to(u.day).value, color='red', linestyle='--', label='Min Distance Time')
ax2.set_ylabel('Distance between S/C and ISO [au]')
ax2.set_title('Distance between Spacecraft and ISO over Time')
ax2.grid(True)

curr_dist, = ax2.plot([], [], color='purple', label='Current Distance', marker='o', linestyle='', markersize=5)
curr_time_2, = ax2.plot([], [], color='black', linestyle='--', zorder=-1)

ax2.set_xlim(sim_params['t_burn'].to(u.day).value, sim_params['max_flight_time'].to(u.day).value)

ax3 = fig.add_subplot(2, 2, 4)

ax3.plot(sc_intercept_ts.to(u.day).value, relative_velocity, color='green')
ax3.axvline(time_of_min_distance.to(u.day).value, color='red', linestyle='--', label='Min Distance Time')
ax3.set_ylabel('Relative Velocity between S/C and ISO [km/s]')
ax3.set_title('Relative Velocity between Spacecraft and ISO over Time')
ax3.grid(True)

curr_vel, = ax3.plot([], [], color='green', label='Current Relative Velocity', marker='o', linestyle='', markersize=5)
curr_time_3, = ax3.plot([], [], color='black', linestyle='--', zorder=-1)

ax3.set_xlabel('Time after ISO Perihelion [days]')

ax3.set_xlim(sim_params['t_burn'].to(u.day).value, sim_params['max_flight_time'].to(u.day).value)

def init():
    sc_intercept_pos.set_data([], [])
    sc_intercept_pos.set_3d_properties([])

    iso_pos.set_data([], [])
    iso_pos.set_3d_properties([])

    curr_dist.set_data([], [])
    curr_vel.set_data([], [])
    curr_time_2.set_data([], [])
    curr_time_3.set_data([], [])

    return sc_intercept_pos, iso_pos, curr_dist, curr_vel, curr_time_2, curr_time_3

def update(frame):
    sc_intercept_pos.set_data([sc_intercept_rs[frame, 0]], [sc_intercept_rs[frame, 1]])
    sc_intercept_pos.set_3d_properties([sc_intercept_rs[frame, 2]])

    iso_pos.set_data([characteristic_iso_states[frame, 0]], [characteristic_iso_states[frame, 1]])
    iso_pos.set_3d_properties([characteristic_iso_states[frame, 2]])

    curr_dist.set_data([sc_intercept_ts.to(u.day).value[frame]], [relative_distance.value[frame]])
    curr_vel.set_data([sc_intercept_ts.to(u.day).value[frame]], [relative_velocity.value[frame]])

    curr_time_2.set_data([sc_intercept_ts.to(u.day).value[frame], sc_intercept_ts.to(u.day).value[frame]],
                         [ax2.get_ylim()[0], ax2.get_ylim()[1]])
    curr_time_3.set_data([sc_intercept_ts.to(u.day).value[frame], sc_intercept_ts.to(u.day).value[frame]],
                         [ax3.get_ylim()[0], ax3.get_ylim()[1]])

    return sc_intercept_pos, iso_pos, curr_dist, curr_vel, curr_time_2, curr_time_3

step = 10
ani = animation.FuncAnimation(fig, update, frames=range(0, len(sc_intercept_ts), step), init_func=init, blit=False, interval=50)

# ani.save('./Trajectory/intercept_trajectory_animation.gif', writer='pillow', fps=30)

plt.show()