import matplotlib.pyplot as plt
import numpy as np
import time as t

from astropy import units as u
from astropy import constants as c
from astropy.coordinates import SkyCoord, solar_system_ephemeris, get_body_barycentric_posvel, CartesianRepresentation, CartesianDifferential
from astropy.time import Time

import matplotlib.animation as animation

import scipy.integrate as integrate
import scipy.optimize as optimize

from poliastro.iod import lambert

from tqdm import tqdm

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['mathtext.fontset'] = 'cm'

# Defining Earth orbital elements
earth_elements = {
    'a': 1 * u.au,
    'e': 0.0167,
    'i': 0 * u.deg,
    'Omega': 0 * u.deg,
    'omega': 102.9372 * u.deg,
    'nu': 0 * u.deg
}

# Defining Mars orbital elements
mars_elements = {
    'a': 1.523679 * u.au,
    'e': 0.0934,
    'i': 1.850 * u.deg,
    'Omega': 49.57854 * u.deg,
    'omega': 336.04084 * u.deg,
    'nu': 0 * u.deg
}

# Defining ISO orbital elements
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

class InterceptorMission:
    def __init__(self, iso_elements, sc_elements, sim_params):
        self.iso_elements = iso_elements
        self.sc_elements = sc_elements
        self.sim_params = sim_params

        # Defining characteristic ISO trajectory
        t_start = self.sim_params['t_burn'].to(u.s).value
        t_end = self.sim_params['max_flight_time'].to(u.s).value
        self.iso_ts = np.linspace(t_start, t_end, 1000)
        self.characteristic_iso_states = np.zeros((len(self.iso_ts), 6))  # Assuming 3 for position and 3 for velocity

        for i, ts in enumerate(self.iso_ts):
            r, v = self.get_iso_state(ts, self.iso_elements)
            self.characteristic_iso_states[i, :3] = r.to(u.au).value
            self.characteristic_iso_states[i, 3:] = v.to(u.km / u.s).value

    def intercept_trajectory(self, verbose=True):
        # Defining time span for intercept propagation
        t_span = np.array([self.sim_params['t_burn'].to(u.day).value, self.sim_params['max_flight_time'].to(u.day).value]) * u.day

        # Initial state of spacecraft at time of burn
        sc_initial_elements = self.sc_elements.copy()
        sc_initial_elements.pop('M', None)
        sc_initial_elements['nu'] = self.sim_params['nu_burn']

        # Rotate dv_chemical_vector to inertial frame
        self.sim_params['dv_chemical_vector'] = (self.R_BI(sc_initial_elements).T @ self.sim_params['dv_chemical_vector'].T).T

        r_initial, v_initial = self._OEtoRV(sc_initial_elements)
        if isinstance(self.sim_params['dv_chemical_vector'], u.Quantity):
            v_initial += self.sim_params['dv_chemical_vector']
        else:
            v_initial += self.sim_params['dv_chemical_vector'] * u.km / u.s

        initial_state = np.concatenate((r_initial.to(u.km).value, v_initial.to(u.km / u.s).value))

        # Propagate intercept trajectory
        sol = self.propagate_intercept(t_span, initial_state, self.sim_params['a_electric'])
        self.sc_intercept_ts = sol.t * u.s
        self.sc_intercept_rs = (sol.y[0:3] * u.km).to(u.au).value.T
        self.sc_intercept_vs = (sol.y[3:6] * u.km / u.s).value.T
        
        # Target position 1000 km ahead of ISO along its velocity vector
        self.target_rs = self.characteristic_iso_states[:, :3] + (1000 * u.km).to(u.au).value * (self.characteristic_iso_states[:, 3:] / np.linalg.norm(self.characteristic_iso_states[:, 3:], axis=1)[:, np.newaxis])

        self.relative_distance = np.linalg.norm(self.sc_intercept_rs - self.target_rs, axis=1) * u.au
        self.relative_velocity = np.linalg.norm(self.sc_intercept_vs - self.characteristic_iso_states[:, 3:], axis=1) * u.km / u.s

        self.min_distance_index = np.argmin(self.relative_distance)

        self.min_distance = self.relative_distance[self.min_distance_index]
        self.rel_v_of_min_distance = self.relative_velocity[self.min_distance_index]
        self.time_of_min_distance = self.sc_intercept_ts[self.min_distance_index]

        self.intercept_time = self.time_of_min_distance - self.sim_params['t_burn']

        if isinstance(self.sim_params['dv_chemical_vector'], u.Quantity):
            dv_chemical_vector = self.sim_params['dv_chemical_vector']
        else:
            dv_chemical_vector = self.sim_params['dv_chemical_vector'] * u.km / u.s
        self.chemical_dv_magnitude = np.linalg.norm(dv_chemical_vector)
        self.electric_dv_magnitude = (self.sim_params["a_electric"] * self.intercept_time).to(u.km / u.s)
        self.total_dv_magnitude = self.chemical_dv_magnitude + self.electric_dv_magnitude

        if verbose:
            print("=== Intercept Trajectory Results ===")
            print(f"Minimum Distance at Intercept: {self.min_distance:.3f}, {self.min_distance.to(u.km):.0f}")
            print(f"Time of Minimum Distance: {self.time_of_min_distance.to(u.day):.2f} after ISO Perihelion")
            print(f"Relative Velocity at Minimum Distance: {self.rel_v_of_min_distance:.6f}")
            print(f'Intercept Trajectory Duration: {self.intercept_time.to(u.day):.0f}')
            print()
            print(f'Required Delta-V for Chemical Boost: {self.chemical_dv_magnitude:.3f}')
            print(f'Delta-V from Electric Populsion: {self.electric_dv_magnitude:.3f}')
            print(f'Total Delta-V: {self.total_dv_magnitude:.3f}')

    def plot_trajectory(self):
        # Generating spacecraft parking orbit data over one orbital period
        t_start = 0
        t_end = self._period(self.sc_elements['a']).to(u.s).value
        self.sc_ts = np.linspace(t_start, t_end, 1000)
        self.sc_states = np.zeros((len(self.sc_ts), 6))

        for i, ts in enumerate(self.sc_ts):
            r_sc, v_sc = self.get_sc_state(ts, self.sc_elements)
            self.sc_states[i, :3] = r_sc.to(u.au).value
            self.sc_states[i, 3:] = v_sc.to(u.km / u.s).value

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        ax.scatter(0, 0, 0, color='orange', s=50, label='Sun', marker='*')

        # ax.plot(earth_states[:, 0], earth_states[:, 1], earth_states[:, 2],
        #         label='Earth Orbit', color='black', linestyle='--')

        # ax.plot(mars_states[:, 0], mars_states[:, 1], mars_states[:, 2],
        #         label='Mars Orbit', color='black', linestyle='--')

        ax.plot(self.characteristic_iso_states[:, 0], self.characteristic_iso_states[:, 1], self.characteristic_iso_states[:, 2],
                label='Characteristic ISO Trajectory', color='red')
        # ax.scatter(self.characteristic_iso_states[-1, 0], self.characteristic_iso_states[-1, 1], self.characteristic_iso_states[-1, 2],
        #            color='red', s=30, label='ISO End', marker='o')
        iso_pos, = ax.plot([], [], [], color='red', marker='o', markersize=8, label='ISO Position', linestyle='')

        ax.plot(self.sc_states[:, 0], self.sc_states[:, 1], self.sc_states[:, 2],
                label='Spacecraft Parking Orbit', color='blue', linestyle=':')

        ax.plot(self.sc_intercept_rs[:, 0], self.sc_intercept_rs[:, 1], self.sc_intercept_rs[:, 2],
                label='Spacecraft Intercept Trajectory', color='blue')
        # ax.scatter(self.sc_intercept_rs[0, 0], self.sc_intercept_rs[0, 1], self.sc_intercept_rs[0, 2],
        #            color='blue', s=30, label='Spacecraft Boost Point', marker='o')
        sc_intercept_pos, = ax.plot([], [], [], color='blue', marker='o', markersize=8, label='S/C Position', linestyle='')

        max_dist = np.abs(self.sc_intercept_rs[:, :3]).max() * 2

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

        ax2.plot(self.sc_intercept_ts.to(u.day).value, self.relative_distance, color='purple')
        ax2.axvline(self.time_of_min_distance.to(u.day).value, color='red', linestyle='--', label='Min Distance Time')
        ax2.set_ylabel('Distance between S/C and ISO [au]')
        ax2.set_title('Distance between Spacecraft and ISO over Time')
        ax2.grid(True)

        curr_dist, = ax2.plot([], [], color='purple', label='Current Distance', marker='o', linestyle='', markersize=5)
        curr_time_2, = ax2.plot([], [], color='black', linestyle='--', zorder=-1)

        ax2.set_xlim(self.sim_params['t_burn'].to(u.day).value, self.sim_params['max_flight_time'].to(u.day).value)

        ax3 = fig.add_subplot(2, 2, 4)

        ax3.plot(self.sc_intercept_ts.to(u.day).value, self.relative_velocity, color='green')
        ax3.axvline(self.time_of_min_distance.to(u.day).value, color='red', linestyle='--', label='Min Distance Time')
        ax3.set_ylabel('Relative Velocity between S/C and ISO [km/s]')
        ax3.set_title('Relative Velocity between Spacecraft and ISO over Time')
        ax3.grid(True)

        curr_vel, = ax3.plot([], [], color='green', label='Current Relative Velocity', marker='o', linestyle='', markersize=5)
        curr_time_3, = ax3.plot([], [], color='black', linestyle='--', zorder=-1)

        ax3.set_xlabel('Time after ISO Perihelion [days]')

        ax3.set_xlim(self.sim_params['t_burn'].to(u.day).value, self.sim_params['max_flight_time'].to(u.day).value)

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
            sc_intercept_pos.set_data([self.sc_intercept_rs[frame, 0]], [self.sc_intercept_rs[frame, 1]])
            sc_intercept_pos.set_3d_properties([self.sc_intercept_rs[frame, 2]])

            iso_pos.set_data([self.characteristic_iso_states[frame, 0]], [self.characteristic_iso_states[frame, 1]])
            iso_pos.set_3d_properties([self.characteristic_iso_states[frame, 2]])
            curr_dist.set_data([self.sc_intercept_ts.to(u.day).value[frame]], [self.relative_distance.value[frame]])
            curr_vel.set_data([self.sc_intercept_ts.to(u.day).value[frame]], [self.relative_velocity.value[frame]])

            curr_time_2.set_data([self.sc_intercept_ts.to(u.day).value[frame], self.sc_intercept_ts.to(u.day).value[frame]],
                                [ax2.get_ylim()[0], ax2.get_ylim()[1]])
            curr_time_3.set_data([self.sc_intercept_ts.to(u.day).value[frame], self.sc_intercept_ts.to(u.day).value[frame]],
                                [ax3.get_ylim()[0], ax3.get_ylim()[1]])

            return sc_intercept_pos, iso_pos, curr_dist, curr_vel, curr_time_2, curr_time_3

        step = 10
        ani = animation.FuncAnimation(fig, update, frames=range(0, len(self.sc_intercept_ts), step), init_func=init, blit=False, interval=50)

        return ani

    def objective_function(self, dv_chemical_vector):
        # Update the chemical delta-v in simulation parameters
        self.sim_params['dv_chemical_vector'] = dv_chemical_vector

        # Recompute the intercept trajectory
        self.intercept_trajectory(verbose=False)

        # Term 1: Distance at closest approach
        # Normalize: let's say 10,000 km is 1 unit
        min_distance = self.min_distance.to(u.au).value
        
        # Term 2: Relative velocity at closest approach
        # Normalize: let's say 10 km/s is 1 unit
        relative_velocity = self.rel_v_of_min_distance.to(u.km / u.s).value
        # # If distance is very small, start optimizing velcity more
        # if min_distance * 10000 < 100000: # 100,000 km
        #     # Identifying weights
        #     weight_distance = 0.1
        #     weight_velocity = 0.9
        # else:
        weight_distance = 0.9
        weight_velocity = 0.1

        return weight_distance * min_distance + weight_velocity * relative_velocity

    def dv_constraint(self, dv_chemical_vector):
        # Constraint: total delta-v should be less than or equal to a specified limit (e.g., 15 km/s)
        dv_limit = 15 * u.km / u.s
        return dv_limit.to(u.km / u.s).value - self.total_dv_magnitude.to(u.km / u.s).value
    
    def solve_intercept_optimization(self):
        # Initial guess for chemical delta-v vector
        dv_initial = self.sim_params['dv_chemical_vector'].value

        self.intercept_trajectory()  # Initial trajectory computation

        # Defining constraints
        constraints = ({
            'type': 'ineq',
            'fun': self.dv_constraint
        })

        # Defining histories
        self.iterations = 0
        self.sc_intercept_ts_history = []
        self.sc_intercept_rs_history = []
        self.sc_intercept_vs_history = []
        self.dv_history = []

        self.min_distance_index_history = []
        self.time_of_min_distance_history = []
        self.min_distance_history = []
        self.rel_v_of_min_distance_history = []

        self.min_distance_index_history.append(self.min_distance_index)
        self.sc_intercept_ts_history.append(self.sc_intercept_ts)
        self.sc_intercept_rs_history.append(self.sc_intercept_rs)
        self.sc_intercept_vs_history.append(self.sc_intercept_vs)
        self.dv_history.append(dv_initial)

        self.time_of_min_distance_history.append(self.time_of_min_distance)
        self.min_distance_history.append(self.min_distance)
        self.rel_v_of_min_distance_history.append(self.rel_v_of_min_distance)

        # Performing optimization using SLSQP method
        print("Starting optimization of chemical delta-v vector...")
        result = optimize.minimize(self.objective_function, 
                                   dv_initial, 
                                   method='SLSQP', 
                                   constraints=constraints,
                                   tol=1e-6,
                                   callback=self.callback_function,
                                   options={'disp': True, 'eps': 1e-7, 'ftol': 1e-6, 'maxiter': 500})

        if result.success:
            optimized_dv = result.x * u.km / u.s
            self.sim_params['dv_chemical_vector'] = optimized_dv
            self.intercept_trajectory(verbose=True)
            anim1 = self.plot_trajectory()
            anim2 = self.plot_history()
            return result, anim1, anim2
        else:
            raise RuntimeError("Optimization failed: " + result.message)
        
    def callback_function(self, xk):
        self.iterations += 1
        self.sc_intercept_ts_history.append(self.sc_intercept_ts)
        self.sc_intercept_rs_history.append(self.sc_intercept_rs)
        self.sc_intercept_vs_history.append(self.sc_intercept_vs)
        self.dv_history.append(xk)
        
        self.min_distance_index_history.append(self.min_distance_index)
        self.time_of_min_distance_history.append(self.time_of_min_distance)
        self.min_distance_history.append(self.min_distance)
        self.rel_v_of_min_distance_history.append(self.rel_v_of_min_distance)

        print(f"Iteration {self.iterations}: Current chemical delta-v magnitude: {np.linalg.norm(xk)} km/s")

    def plot_history(self):
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        ax.scatter(0, 0, 0, color='orange', s=50, label='Sun', marker='*')

        ax.plot(self.characteristic_iso_states[:, 0], self.characteristic_iso_states[:, 1], self.characteristic_iso_states[:, 2],
                label='Characteristic ISO Trajectory', color='red')
        iso_pos, = ax.plot([], [], [], color='red', marker='o', markersize=8, label='ISO Position', linestyle='')
        
        ax.plot(self.sc_states[:, 0], self.sc_states[:, 1], self.sc_states[:, 2],
                label='Spacecraft Parking Orbit', color='blue', linestyle=':')

        sc_intercept, = ax.plot([], [], [], color='blue', label='Spacecraft Intercept Trajectory')
        sc_intercept_pos, = ax.plot([], [], [], color='blue', marker='o', markersize=8, label='S/C Position', linestyle='')

        # max_dist = np.abs(np.array([val.value for val in self.min_distance_history])).max() * 2
        # max_dist = np.abs(self.sc_intercept_rs_history[-1][[self.min_distance_index_history[-1]], :3]).max() * 2
        max_dist = 20

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
        ax2 = fig.add_subplot(3, 2, 2)

        rel_dists, = ax2.plot([], [], color='purple')
        min_time_2, = ax2.plot([], [], color='red', linestyle='--', label='Min Distance Time')

        ax2.set_ylabel('Distance between\nS/C and ISO [au]')
        ax2.set_title('Distance between Spacecraft and ISO over Time')
        ax2.grid(True)
        ax2.set_xlim(self.sim_params['t_burn'].to(u.day).value, self.sim_params['max_flight_time'].to(u.day).value)
        
        ax3 = fig.add_subplot(3, 2, 4)

        rel_vels, = ax3.plot([], [], color='green')
        min_time_3, = ax3.plot([], [], color='red', linestyle='--', label='Min Distance Time')

        ax3.set_ylabel('Relative Velocity between\nS/C and ISO [km/s]')
        ax3.set_title('Relative Velocity between Spacecraft and ISO over Time')
        ax3.grid(True)

        ax3.set_xlabel('Time after ISO Perihelion [days]')
        ax3.set_xlim(self.sim_params['t_burn'].to(u.day).value, self.sim_params['max_flight_time'].to(u.day).value)
        ax3.set_ylim(0, np.amax(self.rel_v_of_min_distance_history[-1]).to(u.km / u.s).value * 1.1)

        ax4 = fig.add_subplot(3, 2, 6)

        ax4.plot(range(len(self.dv_history)), [np.linalg.norm(dv) for dv in self.dv_history], color='brown')
        curr_iter, = ax4.plot([], [], color='red', linestyle='--', label='Current Iteration')

        ax4.set_xlabel('Optimization Iteration')
        ax4.set_ylabel('Chemical Delta-V\nMagnitude [km/s]')
        ax4.set_title('Chemical Delta-V Magnitude over Optimization Iterations')
        ax4.grid(True)

        ax4.set_ylim(0, np.linalg.norm(self.dv_history[-1]) * 1.1)

        def init():
            sc_intercept.set_data([], [])
            sc_intercept.set_3d_properties([])

            sc_intercept_pos.set_data([], [])
            sc_intercept_pos.set_3d_properties([])

            iso_pos.set_data([], [])
            iso_pos.set_3d_properties([])

            rel_dists.set_data([], [])
            rel_vels.set_data([], [])
            min_time_2.set_data([], [])
            min_time_3.set_data([], [])

            curr_iter.set_data([], [])

            return sc_intercept, sc_intercept_pos, iso_pos, rel_dists, rel_vels, min_time_2, min_time_3, curr_iter
        
        def update(frame):
            fig.suptitle(f'Interception Time since ISO Perihelion: {self.scanner_times_array[frame].to(u.day).value:.2f} days', fontsize=16, fontweight='bold')

            sc_intercept.set_data(self.sc_intercept_rs_history[frame][:, 0], self.sc_intercept_rs_history[frame][:, 1])
            sc_intercept.set_3d_properties(self.sc_intercept_rs_history[frame][:, 2])

            sc_intercept_pos.set_data([self.sc_intercept_rs_history[frame][self.min_distance_index_history[frame], 0]],
                                      [self.sc_intercept_rs_history[frame][self.min_distance_index_history[frame], 1]])
            sc_intercept_pos.set_3d_properties([self.sc_intercept_rs_history[frame][self.min_distance_index_history[frame], 2]])

            iso_pos.set_data([self.characteristic_iso_states[self.min_distance_index_history[frame], 0]],
                             [self.characteristic_iso_states[self.min_distance_index_history[frame], 1]])
            iso_pos.set_3d_properties([self.characteristic_iso_states[self.min_distance_index_history[frame], 2]])

            rel_dists.set_data([self.sc_intercept_ts_history[frame].to(u.day).value],
                               [np.linalg.norm(self.sc_intercept_rs_history[frame] - self.characteristic_iso_states[:, :3], axis=1)])

            rel_vels.set_data([self.sc_intercept_ts_history[frame].to(u.day).value],
                              [np.linalg.norm(self.sc_intercept_vs_history[frame] - self.characteristic_iso_states[:, 3:], axis=1)])
            
            ax2.set_ylim(0, np.linalg.norm(self.sc_intercept_rs_history[frame] - self.characteristic_iso_states[:, :3], axis=1).max() * 1.1)

            ax3.set_ylim(0, np.linalg.norm(self.sc_intercept_vs_history[frame] - self.characteristic_iso_states[:, 3:], axis=1).max() * 1.1)

            min_time_2.set_data([self.time_of_min_distance_history[frame].to(u.day).value, self.time_of_min_distance_history[frame].to(u.day).value],
                                [ax2.get_ylim()[0], ax2.get_ylim()[1]])
            min_time_3.set_data([self.time_of_min_distance_history[frame].to(u.day).value, self.time_of_min_distance_history[frame].to(u.day).value],
                                [ax3.get_ylim()[0], ax3.get_ylim()[1]])
            
            curr_iter.set_data([frame, frame], 
                               [ax4.get_ylim()[0], ax4.get_ylim()[1]])

            return sc_intercept, sc_intercept_pos, iso_pos, rel_dists, rel_vels, min_time_2, min_time_3, curr_iter
        
        ani = animation.FuncAnimation(fig, update, frames=range(len(self.sc_intercept_ts_history)), init_func=init, blit=False, interval=50)

        return ani

    def solve_intercept_lambert(self, verbose=True):
        max_dist = self.sim_params['max_distance']

        # Calculating time of flight of ISO to reach max distance from the Sun
        within_dist_mask = np.linalg.norm(self.characteristic_iso_states[:, :3], axis=1) * u.au <= max_dist
        if np.all(within_dist_mask):
            raise ValueError("ISO does not reach the specified maximum distance within the simulated trajectory.")
        
        max_time = self.iso_ts[within_dist_mask][-1] * u.s
        self.sim_params['max_flight_time'] = max_time

        # Defining characteristic ISO trajectory
        t_start = self.sim_params['t_burn'].to(u.s).value
        t_end = self.sim_params['max_flight_time'].to(u.s).value
        self.iso_ts = np.linspace(t_start, t_end, 1000)
        self.characteristic_iso_states = np.zeros((len(self.iso_ts), 6))  # Assuming 3 for position and 3 for velocity

        for i, ts in enumerate(self.iso_ts):
            r, v = self.get_iso_state(ts, self.iso_elements)
            self.characteristic_iso_states[i, :3] = r.to(u.au).value
            self.characteristic_iso_states[i, 3:] = v.to(u.km / u.s).value

        # Initial state of spacecraft at time of burn
        sc_initial_elements = self.sc_elements.copy()
        sc_initial_elements.pop('M', None)
        sc_initial_elements['nu'] = self.sim_params['nu_burn']

        r_initial, v_initial = self._OEtoRV(sc_initial_elements)

        # Final position of ISO at max distance
        r_final = self.characteristic_iso_states[-1, :3] * u.au

        # Solving Lambert's problem to get required velocity at burn
        (v_initial_lambert, v_final_lambert) = lambert(c.GM_sun, r_initial, r_final, (t_end - t_start) * u.s)

        dv_chemical_vector = (v_initial_lambert - v_initial).to(u.km / u.s)
        intercept_relative_velocity = np.linalg.norm((v_final_lambert.to(u.km / u.s).value - self.characteristic_iso_states[-1, 3:])) * u.km / u.s

        # Transform dv_chemical_vector to body frame
        dv_chemical_vector = (self.R_BI(sc_initial_elements) @ dv_chemical_vector.T).T

        if verbose:
            print("=== Lambert Intercept Results ===")
            print(f"Chemical Delta-V Vector Required at Burn: {dv_chemical_vector}")
            print(f"Chemical Delta-V Magnitude Required at Burn: {np.linalg.norm(dv_chemical_vector):.3f}")
            print(f"Relative Velocity at Intercept: {intercept_relative_velocity:.6f}")
            print()

        self.sim_params['dv_chemical_vector'] = dv_chemical_vector

    def solve_intercept_lambert_scanner(self, verbose=True, animate=True, steps=10):
        max_dist = self.sim_params['max_distance']

        # Calculating time of flight of ISO to reach max distance from the Sun
        within_dist_mask = np.linalg.norm(self.characteristic_iso_states[:, :3], axis=1) * u.au <= max_dist
        if np.all(within_dist_mask):
            raise ValueError("ISO does not reach the specified maximum distance within the simulated trajectory.")
        
        max_time = self.iso_ts[within_dist_mask][-1] * u.s
        self.sim_params['max_flight_time'] = max_time

        # Defining characteristic ISO trajectory
        t_start = self.sim_params['t_burn'].to(u.s).value
        t_end = self.sim_params['max_flight_time'].to(u.s).value
        self.iso_ts = np.linspace(t_start, t_end, 1000)
        self.characteristic_iso_states = np.zeros((len(self.iso_ts), 6))  # Assuming 3 for position and 3 for velocity

        for i, ts in enumerate(self.iso_ts):
            r, v = self.get_iso_state(ts, self.iso_elements)
            self.characteristic_iso_states[i, :3] = r.to(u.au).value
            self.characteristic_iso_states[i, 3:] = v.to(u.km / u.s).value

        # Initial state of spacecraft at time of burn
        sc_initial_elements = self.sc_elements.copy()
        sc_initial_elements.pop('M', None)
        sc_initial_elements['nu'] = self.sim_params['nu_burn']

        r_initial, v_initial = self._OEtoRV(sc_initial_elements)

        # Defining histories
        self.iterations = 0
        self.scanner_times = []

        self.sc_intercept_ts_history = []
        self.sc_intercept_rs_history = []
        self.sc_intercept_vs_history = []
        self.dv_history = []

        self.min_distance_index_history = []
        self.time_of_min_distance_history = []
        self.min_distance_history = []
        self.rel_v_of_min_distance_history = []

        # Starting scan at final position of ISO to first position at ISO
        for idx in tqdm(range(len(self.characteristic_iso_states)-1, 0, -steps), disable=not verbose):
            self.scanner_times.append(self.iso_ts[idx])

            # Final position of ISO at current index
            r_final = self.characteristic_iso_states[idx, :3] * u.au
            
            # Time of flight to reach that position
            t_flight = (self.iso_ts[idx] - t_start) * u.s

            # Final velocity of ISO at current index
            v_final_iso = self.characteristic_iso_states[idx, 3:] * u.km / u.s

            # We want the spacecraft to meet the ISO at this time but 1000 km ahead along its velocity vector
            # r_final += (1000 * u.km).to(u.au) * (v_final_iso / np.linalg.norm(v_final_iso))

            # Solving Lambert's problem to get required velocity at burn
            try:
                (v_initial_lambert, v_final_lambert) = lambert(c.GM_sun, r_initial, r_final, t_flight, rtol=1e-12)
            except:
                continue

            dv_chemical_vector = (v_initial_lambert - v_initial).to(u.km / u.s)
            intercept_relative_velocity = np.linalg.norm((v_final_lambert.to(u.km / u.s).value - self.characteristic_iso_states[idx, 3:])) * u.km / u.s

            # Transform dv_chemical_vector to body frame
            dv_chemical_vector = (self.R_BI(sc_initial_elements) @ dv_chemical_vector.T).T

            # self.sim_params['max_flight_time'] = self.iso_ts[idx] * u.s
            self.sim_params['dv_chemical_vector'] = dv_chemical_vector

            self.intercept_trajectory(verbose=False)
            
            self.sc_intercept_ts_history.append(self.sc_intercept_ts)
            self.sc_intercept_rs_history.append(self.sc_intercept_rs)
            self.sc_intercept_vs_history.append(self.sc_intercept_vs)
            self.dv_history.append(dv_chemical_vector.value)
            
            self.min_distance_index_history.append(self.min_distance_index)
            self.time_of_min_distance_history.append(self.time_of_min_distance)
            self.min_distance_history.append(self.min_distance)
            self.rel_v_of_min_distance_history.append(self.rel_v_of_min_distance)

            self.iterations += 1

        self.min_dv_index = np.argmin([np.linalg.norm(dv) for dv in self.dv_history])

        if verbose:
            print("=== Lambert Scan Results ===")
            print(f"Optimal Chemical Delta-V Vector Required at Burn: {self.dv_history[self.min_dv_index]} km/s")
            print(f"Optimal Chemical Delta-V Magnitude Required at Burn: {np.linalg.norm(self.dv_history[self.min_dv_index]):.3f} km/s")
            print(f"Minimum Distance at Intercept: {self.min_distance_history[self.min_dv_index]:.3f}, {self.min_distance_history[self.min_dv_index].to(u.km):.0f}")
            print(f"Time of Minimum Distance: {self.time_of_min_distance_history[self.min_dv_index].to(u.day):.2f} after ISO Perihelion")
        
        self.sim_params['dv_chemical_vector'] = self.dv_history[self.min_dv_index] * u.km / u.s
        
        self.intercept_trajectory(verbose=False)

        if animate:
            anim1 = self.plot_trajectory()
            anim2 = self.plot_history()
        
        else:
            anim1 = None
            anim2 = None
        
        self.scanner_times_array = np.array(self.scanner_times) * u.s
        
        self.sc_intercept_ts_history_array = np.array([val.to(u.s).value for val in self.sc_intercept_ts_history]) * u.s
        self.sc_intercept_rs_history_array = np.array(self.sc_intercept_rs_history) * u.au
        self.sc_intercept_vs_history_array = np.array(self.sc_intercept_vs_history) * u.km / u.s
        self.dv_history_array = np.array(self.dv_history) * u.km / u.s

        self.min_distance_index_history_array = np.array(self.min_distance_index_history)
        self.time_of_min_distance_history_array = np.array([val.to(u.s).value for val in self.time_of_min_distance_history]) * u.s
        self.min_distance_history_array = np.array([val.to(u.km).value for val in self.min_distance_history]) * u.km
        self.rel_v_of_min_distance_history_array = np.array([val.to(u.km / u.s).value for val in self.rel_v_of_min_distance_history]) * u.km / u.s

        return anim1, anim2


    def _hyperbolic_mean_to_eccentric_anomaly(self, M, e):
        '''
        Convert hyperbolic mean anomaly to hyperbolic eccentric anomaly using Newton-Raphson method:
        
        `M = e * sinh(F) - F`

        Parameters
        ----------
        M : float
            Hyperbolic mean anomaly in radians
        e : float
            Eccentricity (e > 1 for hyperbolic orbits)

        Returns
        -------
        F : float
            Hyperbolic eccentric anomaly in radians
        '''
        if e <= 1:
            raise ValueError("Eccentricity must be greater than 1 for hyperbolic orbits.")

        def func(F):
            return e * np.sinh(F) - F - M
        def deriv(F):
            return e * np.cosh(F) - 1
        
        # Initial guess: F approx M for small M, else log(2M/e)
        F0 = M if abs(M) < 1 else np.log(2 * abs(M) / e) * np.sign(M)

        # Performing Newton-Raphson using scipy.optimize
        F = optimize.newton(func, F0, fprime=deriv, tol=1e-10, maxiter=1000)

        # # Newton-Raphson iterations
        # for k in range(max_iter):
        #     # Calculating next correction
        #     delta_F = (M - (e * np.sinh(F) - F)) / (e * np.cosh(F) - 1)
            
        #     # Calculating next approximation
        #     F += delta_F

        #     # Check for convergence
        #     if abs(delta_F) < tol:
        #         break

        # return F, k+1  # Return the number of iterations as well
        
        return F

    def _mean_to_eccentric_anomaly(self, M, e):
        '''
        Convert mean anomaly to eccentric anomaly using Newton-Raphson method:
        
        `M = E - e * sin(E)`

        Parameters
        ----------
        M : float
            Mean anomaly in radians
        e : float
            Eccentricity (e < 1 for elliptical orbits)

        Returns
        -------
        E : float
            Eccentric anomaly in radians
        '''
        if e >= 1:
            raise ValueError("Eccentricity must be less than 1 for elliptical orbits.")
        
        def func(E):
            return E - e * np.sin(E) - M
        def deriv(E):
            return 1 - e * np.cos(E)
        
        # Initial guess: E approx M for small e, else pi
        E0 = np.pi if e > 0.8 else M 

        # Performing Newton-Raphson using scipy.optimize
        E = optimize.newton(func, E0, fprime=deriv)

        return E

    def _OEtoRV(self, orbital_elements: dict) -> dict:
        '''
        Convert orbital elements to state vectors.

        Parameters
        ----------
        orbital_elements : dict
            Dictionary containing heliocentric orbital elements:
            - a: Semi-major axis (astropy Quantity with length unit)
            - e: Eccentricity (float)
            - i: Inclination (astropy Quantity with angle unit)
            - Omega: Longitude of ascending node (astropy Quantity with angle unit)
            - omega: Argument of perihelion (astropy Quantity with angle unit)
            - M: Mean anomaly (astropy Quantity with angle unit)
            - epoch: Epoch of the elements (astropy Time object)

        Returns
        -------
        (r, v) : tuple of np.ndarray
            Position (r) and velocity (v) vectors in heliocentric frame [km] and [km/s]
        '''
        # Unpacking orbital elements and converting to appropriate units
        a = orbital_elements['a'].to(u.km).value
        e = orbital_elements['e']
        i = orbital_elements['i'].to(u.rad).value
        Omega = orbital_elements['Omega'].to(u.rad).value
        omega = orbital_elements['omega'].to(u.rad).value

        # Checking if mean anomaly or true anomaly is given:
        if 'M' in orbital_elements:
            M = orbital_elements['M'].to(u.rad).value
        elif 'nu' in orbital_elements:
            nu = orbital_elements['nu'].to(u.rad).value
        else:
            raise ValueError("Either mean anomaly (M) or true anomaly (nu) must be provided.")

        # Defining constants
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]
        h = np.sqrt(abs(mu_sun * a * (e**2 - 1)))  # Specific angular momentum [km^2/s]

        # Defining rotational matrices
        Rz_Omega = np.array([
            [ np.cos(Omega), np.sin(Omega), 0.0],
            [-np.sin(Omega), np.cos(Omega), 0.0],
            [0.0,         0.0,        1.0]
        ])

        Rx_i = np.array([
            [1.0,  0.0,       0.0],
            [0.0,  np.cos(i), np.sin(i)],
            [0.0, -np.sin(i), np.cos(i)]
        ])

        Rz_omega = np.array([
            [ np.cos(omega), np.sin(omega), 0.0],
            [-np.sin(omega), np.cos(omega), 0.0],
            [0.0,         0.0,        1.0]
        ])

        # DCM to go from inertial (heliocentric) frame to orbital frame
        R_OI = Rz_omega @ Rx_i @ Rz_Omega

        # Calculating position of object in the orbital frame
        # Solving Kepler's equation for Eccentric Anomaly (E) using Newton-Raphson method
        if 'M' in orbital_elements:
            if e < 1:
                # Elliptical orbit case
                E = self._mean_to_eccentric_anomaly(M, e)

                # Calculating true anomaly (nu) from eccentric anomaly (E)
                nu = 2 * np.arctan2( np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2), 1 )

                # Calculating distance (r) from the focus (Sun) to the object
                r_mag = (a * (1 - e**2)) / (1 + e * np.cos(nu))

            elif e > 1:
                # Hyperbolic orbit case
                F = self._hyperbolic_mean_to_eccentric_anomaly(M, e)

                # Calculating true anomaly (nu) from hyperbolic eccentric anomaly (F)
                nu = 2 * np.arctan2( np.sqrt((e + 1) / (e - 1)) * np.tanh(F / 2), 1 )

                # Calculating distance (r) from the focus (Sun) to the object
                r_mag = (a * (1 - e**2)) / (1 + e * np.cos(nu))

            else:
                raise ValueError("Eccentricity cannot be equal to 1 (parabolic orbits not supported).")
        
        else:
            # If true anomaly is given directly
            r_mag = (a * (1 - e**2)) / (1 + e * np.cos(nu))

        # Position in orbital frame [km]
        r_orbital = np.array([
            r_mag * np.cos(nu),
            r_mag * np.sin(nu),
            0.0
        ])

        # Velocity in orbital frame [km/s]
        v_orbital = np.array([
            (mu_sun / h) * (-np.sin(nu)),
            (mu_sun / h) * (e + np.cos(nu)),
            0.0
        ])

        # Rotating position and velocity to heliocentric frame
        r_heliocentric = (np.transpose(R_OI) @ r_orbital * u.km).to(u.au)
        v_heliocentric = np.transpose(R_OI) @ v_orbital * u.km / u.s

        return (r_heliocentric, v_heliocentric)

    def R_OI(self, orbital_elements: dict) -> dict:
        '''
        Computes rotation matrix from inertial frame to orbital frame.

        Parameters
        ----------
        orbital_elements : dict
            Dictionary containing heliocentric orbital elements:
            - a: Semi-major axis (astropy Quantity with length unit)
            - e: Eccentricity (float)
            - i: Inclination (astropy Quantity with angle unit)
            - Omega: Longitude of ascending node (astropy Quantity with angle unit)
            - omega: Argument of perihelion (astropy Quantity with angle unit)
            - M: Mean anomaly (astropy Quantity with angle unit)
            - epoch: Epoch of the elements (astropy Time object)

        Returns
        -------
        R_OI : NDArray
            Direction cosine matrix from inertial to orbital frame
        '''
        # Unpacking orbital elements and converting to appropriate units
        a = orbital_elements['a'].to(u.km).value
        e = orbital_elements['e']
        i = orbital_elements['i'].to(u.rad).value
        Omega = orbital_elements['Omega'].to(u.rad).value
        omega = orbital_elements['omega'].to(u.rad).value

        # Checking if mean anomaly or true anomaly is given:
        if 'M' in orbital_elements:
            M = orbital_elements['M'].to(u.rad).value
        elif 'nu' in orbital_elements:
            nu = orbital_elements['nu'].to(u.rad).value
        else:
            raise ValueError("Either mean anomaly (M) or true anomaly (nu) must be provided.")

        # Defining constants
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]
        h = np.sqrt(abs(mu_sun * a * (e**2 - 1)))  # Specific angular momentum [km^2/s]

        # Defining rotational matrices
        Rz_Omega = np.array([
            [ np.cos(Omega), np.sin(Omega), 0.0],
            [-np.sin(Omega), np.cos(Omega), 0.0],
            [0.0,         0.0,        1.0]
        ])

        Rx_i = np.array([
            [1.0,  0.0,       0.0],
            [0.0,  np.cos(i), np.sin(i)],
            [0.0, -np.sin(i), np.cos(i)]
        ])

        Rz_omega = np.array([
            [ np.cos(omega), np.sin(omega), 0.0],
            [-np.sin(omega), np.cos(omega), 0.0],
            [0.0,         0.0,        1.0]
        ])

        # DCM to go from inertial (heliocentric) frame to orbital frame
        R_OI = Rz_omega @ Rx_i @ Rz_Omega

        return R_OI

    def R_BI(self, sc_elements):
        '''
        Computes rotation matrix from inertial frame to spacecraft body frame.

        Parameters
        ----------
        orbital_elements : dict
            Dictionary containing heliocentric orbital elements:
            - a: Semi-major axis (astropy Quantity with length unit)
            - e: Eccentricity (float)
            - i: Inclination (astropy Quantity with angle unit)
            - Omega: Longitude of ascending node (astropy Quantity with angle unit)
            - omega: Argument of perihelion (astropy Quantity with angle unit)
            - M: Mean anomaly (astropy Quantity with angle unit)
            - epoch: Epoch of the elements (astropy Time object)

        Returns
        -------
        R_BI : NDArray
            Direction cosine matrix from inertial to spacecraft body frame
        '''
        # Calculating heliocentric state vectors
        r, v = self._OEtoRV(sc_elements)

        # Calculating velocity unit vector (along +x body axis)
        v_unit = v.value / np.linalg.norm(v.value)

        # Calculating radial vector (along +y body axis)
        r_unit = r.value / np.linalg.norm(r.value)

        # Calculating angular momentum vector (along +z body axis)
        h_unit = np.cross(r_unit, v_unit)
        h_unit /= np.linalg.norm(h_unit)

        # Constructing DCM from inertial to body frame
        R_BI = np.vstack((v_unit, r_unit, h_unit))

        return R_BI
        
    def _helio_RV_to_bary_RV(self, r_heliocentric, v_heliocentric, M, a, perihelion_epoch):
        '''
        Convert heliocentric state vectors to barycentric state vectors.

        Parameters
        ----------
        r_heliocentric : np.ndarray
            Position vector in heliocentric frame [km]
        v_heliocentric : np.ndarray
            Velocity vector in heliocentric frame [km/s]
        a : astropy Quantity with length unit
            Heliocentric semi-major axis
        M : astropy Quantity with angle unit
            Mean anomaly
        perihelion_epoch : astropy Time object
            Epoch of the perihelion passage

        Returns
        -------
        (r, v) : tuple of np.ndarray
            Position (r) and velocity (v) vectors in barycentric frame [km] and [km/s]
        '''
        # Defining constants
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]

        # Converting values
        a = a.to(u.km).value
        M = M.to(u.rad).value

        # Now, we must find what time these elements were valid at, and get the Sun's state vector at that time
        # The epoch given is the perhelion passage time, so we need to propagate to the time of interest
        # First, calculate mean motion (n)
        n = np.sqrt(mu_sun / abs(a)**3)  # Mean motion [rad/s]

        # Now, using M = n * (t - perihelion_epoch), we can find the time of interest
        # Rearranging gives t = M/n + perihelion_epoch
        t = (M / n) * u.s + perihelion_epoch

        # The time t is the time at which the orbital elements are valid
        # We will use this time to get the Sun's state vector
        with solar_system_ephemeris.set('jpl'):
            sun_posvel = get_body_barycentric_posvel('sun', t)
        
        sun_pos = sun_posvel[0].get_xyz().to(u.km)
        sun_vel = sun_posvel[1].get_xyz().to(u.km / u.s)

        # Transforming to barycentric frame
        r_barycentric = r_heliocentric + sun_pos
        v_barycentric = v_heliocentric + sun_vel

        return (r_barycentric, v_barycentric)

    def _RVtoOE(self, r_barycentric, v_barycentric):
        '''
        Convert barycentric state vectors to barycentric orbital elements.

        Parameters
        ----------
        r_barycentric : np.ndarray
            Position vector in barycentric frame [km]
        v_barycentric : np.ndarray
            Velocity vector in barycentric frame [km/s]

        Returns
        -------
        orbital_elements : dict
            Dictionary containing heliocentric orbital elements:
            - a: Semi-major axis (astropy Quantity with length unit)
            - e: Eccentricity (float)
            - i: Inclination (astropy Quantity with angle unit)
            - Omega: Longitude of ascending node (astropy Quantity with angle unit)
            - omega: Argument of perihelion (astropy Quantity with angle unit)
            - M: Mean anomaly (astropy Quantity with angle unit)
            - epoch: Epoch of the elements (astropy Time object)
        '''
        r_barycentric = r_barycentric.to(u.km).value
        v_barycentric = v_barycentric.to(u.km / u.s).value

        # Defining constants
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]

        # Now, we need to convert the barycentric state vector back to barycentric orbital elements
        r_mag_barycentric = np.linalg.norm(r_barycentric)
        v_mag_barycentric = np.linalg.norm(v_barycentric)

        # Specific angular momentum vector
        h_barycentric = np.cross(r_barycentric, v_barycentric)
        h_mag_barycentric = np.linalg.norm(h_barycentric)

        # Calculating semimajor axis (a) [km]
        a = 1 / (2 / np.linalg.norm(r_barycentric) - np.linalg.norm(v_barycentric)**2 / mu_sun)

        # Calculating eccentricity (e)
        e_vector = (1/mu_sun) * (np.cross(v_barycentric, h_barycentric) - mu_sun*r_barycentric/r_mag_barycentric)
        e = np.linalg.norm(e_vector)

        # Calculating orbital frame basis vectors
        i_e = e_vector / e  # Unit vector in direction of eccentricity
        i_h = h_barycentric / h_mag_barycentric  # Unit vector in direction of angular momentum
        i_y = np.cross(i_h, i_e)  # Unit vector perpendicular to both

        # Constructing DCM from inertial to orbital frame
        R_OI_barycentric = np.vstack((i_e, i_y, i_h))

        # Calculating inclination (i) [rad]
        i = np.arccos(R_OI_barycentric[2, 2])

        # Calculating longitude of ascending node (Omega) [rad]
        Omega = np.arctan2(i_h[0]/np.sin(i), -i_h[1]/np.sin(i))

        if Omega < 0:
            Omega += 2 * np.pi

        # Calculating argument of perihelion (omega) [rad]
        omega = np.arctan2(i_e[2]/np.sin(i), i_y[2]/np.sin(i))

        if omega < 0:
            omega += 2 * np.pi

        # Calculating true anomaly (nu) [rad]
        r_orbital = R_OI_barycentric @ r_barycentric
        nu = np.arctan2(r_orbital[1], r_orbital[0])

        # Calculating hyperbolic eccentric anomaly (F) [rad]
        F = 2 * np.arctanh( np.tan(nu / 2) * np.sqrt((e - 1) / (e + 1)) )

        # Calculating hyperbolic mean anomaly (M) [rad]
        M = e * np.sinh(F) - F

        # Inserting values into dictionary and converting to appropriate units
        orbital_elements = {
            'a': (a * u.km).to(u.au),
            'e': e,
            'i': (i * u.rad).to(u.deg),
            'Omega': (Omega * u.rad).to(u.deg),
            'omega': (omega * u.rad).to(u.deg),
            'M': (M * u.rad).to(u.deg),
        }

        return orbital_elements

    def _OEtoRAD(self, a: u.Quantity, e: float, i_deg: u.Quantity, Omega_deg: u.Quantity, omega_deg: u.Quantity, is_barycentric=False) -> tuple[float, float, np.ndarray, np.ndarray]:
        '''
        Convert orbital elements to RA and Dec of the incoming velocity vector if an ISO.

        Parameters
        ----------
        a : float
            Semi-major axis of the orbit (in AU).
        e : float
            Eccentricity of the orbit.
        i_deg : float
            Inclination of the orbit (in degrees).
        Omega_deg : float
            Longitude of the ascending node (in degrees).
        omega_deg : float
            Argument of perihelion (in degrees).
        is_barycentric : bool
            If True, convert to barycentric coordinates.

        Returns
        -------
        ra : float
            Right Ascension of the incoming velocity vector (in degrees).
        dec : float
            Declination of the incoming velocity vector (in degrees).
        '''
        # Defining constants
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]

        # Convert angles from degrees to radians
        i = i_deg.to(u.rad).value
        Omega = Omega_deg.to(u.rad).value
        omega = omega_deg.to(u.rad).value

        # Convert semi-major axis to meters
        a = a.to(u.km).value

        # Hyperbola: e > 1, a < 0
        # Asymptote true anomaly magnitude:
        # cos(nu_inf) = -1/e
        nu_inf = np.arccos(-1.0 / e)      # in radians
        nu_in = -nu_inf                   # incoming branch (before perihelion)

        # Specific angular momentum (magnitude) h = sqrt(mu * a * (e^2 - 1))
        h = np.sqrt(-mu_sun * abs(a) * (1.0 - e**2))

        # Distance at true anomaly nu:
        r = (a * (1.0 - e**2)) / (1.0 + e * np.cos(nu_in))   # km
        print(1 + e * np.cos(nu_in))
        
        # Radial and transverse velocities in perifocal frame:
        v_r = (mu_sun / h) * e * np.sin(nu_in)
        v_t = (mu_sun / h) * (1.0 + e * np.cos(nu_in))

        # Position in perifocal frame
        r_pf = np.array([r * np.cos(nu_in), r * np.sin(nu_in), 0.0])

        # Velocity in perifocal frame
        v_pf = np.array([v_r * np.cos(nu_in) - v_t * np.sin(nu_in),
                        v_r * np.sin(nu_in) + v_t * np.cos(nu_in),
                        0.0])
        
        # Rotation matrix from perifocal -> ECI (ICRS/ecliptic depending on elements)
        # R = R_z(Omega) @ R_x(i) @ R_z(omega) (3-1-3 rotation)
        Rz_Omega = np.array([
            [np.cos(Omega), -np.sin(Omega), 0.0],
            [np.sin(Omega),  np.cos(Omega), 0.0],
            [0.0,         0.0,        1.0]
        ])

        Rx_i = np.array([
            [1.0, 0.0,        0.0],
            [0.0, np.cos(i), -np.sin(i)],
            [0.0, np.sin(i),  np.cos(i)]
        ])

        Rz_omega = np.array([
            [np.cos(omega), -np.sin(omega), 0.0],
            [np.sin(omega),  np.cos(omega), 0.0],
            [0.0,         0.0,        1.0]
        ])

        R = Rz_Omega @ Rx_i @ Rz_omega

        r_eci = R @ r_pf   # km
        v_eci = R @ v_pf   # km/s

        # Incoming velocity vector is v_eci (points toward Sun for incoming object).
        # Radiant is opposite direction (the direction it came from)
        v_unit = v_eci / np.linalg.norm(v_eci)
        radiant_unit = -v_unit

        # Convert to RA, Dec (ICRS) using cartesian -> skycoord
        # Coordinates expect units; astropy's SkyCoord.from_cartesian wants cartesian representation.
        sc = SkyCoord(x=radiant_unit[0], y=radiant_unit[1], z=radiant_unit[2],
                    representation_type='cartesian', frame='icrs')
        
        ra = sc.spherical.lon.deg
        dec = sc.spherical.lat.deg
        
        return ra * u.deg, dec * u.deg, v_eci * u.km / u.s, r_eci * u.km

    def _period(self, a):
        '''
        Calculate the orbital period given the semi-major axis.

        Parameters
        ----------
        a : astropy Quantity with length unit
            Semi-major axis of the orbit.

        Returns
        -------
        T : astropy Quantity with time unit
            Orbital period.
        '''
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]
        a_km = a.to(u.km).value

        T = 2 * np.pi * np.sqrt(a_km**3 / mu_sun) * u.s

        return T

    def get_iso_state(self, t, iso_elements: dict):
        '''
        Returns r, v of ISO at time t (seconds from perihelion).
        '''
        # Defining constants
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]

        a = iso_elements['a'].to(u.km).value # Should be negative for hyperbola
        e = iso_elements['e']
        n = np.sqrt(mu_sun / -a**3) # Mean motion for hyperbola
        
        M = n * t # Mean Anomaly
        iso_elements['M'] = M * u.rad
        
        return self._OEtoRV(iso_elements)

    def get_sc_state(self, t, sc_elements: dict):
        '''
        Returns r, v of SC at time t (seconds from perihelion).
        '''
        # Defining constants
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]

        a = sc_elements['a'].to(u.km).value
        e = sc_elements['e']
        n = np.sqrt(mu_sun / a**3) # Mean motion for ellipse
        
        M = n * t # Mean Anomaly
        sc_elements['M'] = M * u.rad
        
        return self._OEtoRV(sc_elements)

    def equations_of_motion(self, t, x, a_electric):
        '''
        Calculates the derivatives of the state vector x at time t due to both solar gravity and electric thrust.

        Parameters
        ----------
        t : float
            Current time [s]
        x : NDArray
            Current state vector (1x6)
        a_electric : NDArray
            Electric thrust acceleration vector (1x3)

        Returns
        -------
        x_dot : NDArray
            Derivatives of the state vector (1x6)
        '''
        state = x

        # Defining constants
        mu_sun = c.GM_sun.to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]

        # Unpack the state vector
        r, v = state[:3], state[3:]

        # Magnitude of the position vector
        r_mag = np.linalg.norm(r)

        # Magnitude of the velocity vector
        v_mag = np.linalg.norm(v)

        # Calculate the gravitational acceleration
        a_grav = -mu_sun / r_mag**3 * r

        # Calculate the electric thrust acceleration (in the direction of motion)
        a_electric = a_electric.to(u.km / u.s**2).value * (v / v_mag)

        # Total acceleration
        a = a_grav + a_electric

        state_dot = np.concatenate((v, a))

        # Return the derivatives
        return state_dot

    def propagate_intercept(self, t_span, initial_state, a_electric, target_func=None):
        '''
        Integrates the spacecraft trajectory under solar gravity and electric thrust.

        Parameters
        ----------
        t_span : NDArray
            Time span for integration [s]
        initial_state : NDArray
            Initial state vector (1x6)
        a_electric : NDArray
            Electric thrust acceleration vector (1x3)
        target_func : function
            Function that returns the target state vector at time t

        Returns
        -------
        sol : OdeResult
            Solution object from scipy.integrate.solve_ivp
        '''
        # Time array for integration
        ts = np.linspace(t_span[0].to(u.s).value, t_span[1].to(u.s).value, 1000)

        # Integrating the equations of motion using scipy.integrate.solve_ivp
        sol = integrate.solve_ivp(self.equations_of_motion, [t_span[0].to(u.s).value, t_span[1].to(u.s).value], initial_state, t_eval=ts, args=(a_electric.to(u.km / u.s**2),), rtol=1e-6, atol=1e-6)

        return sol

    def analyze_interception_scenario(self, iso_elements: dict, sc_parking_elements: dict, sim_params: dict):
        '''
        Determines interception details between a spacecraft and an interstellar object (ISO).
        '''
        # Defining constants
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]

        # 1. Setup Timing
        # t=0 is ISO perihelion.
        t_detect = sim_params['t_detect'] # e.g., -50 days (seconds)
        t_burn = sim_params['t_burn']     # Time S/C executes chemical burn

        # 2. State of ISO at Burn Time (to help target the chemical burn)
        r_iso_burn, v_iso_burn = self.get_iso_state(t_burn, iso_elements)
        
        # 3. State of S/C at Burn Time
        # Assume circular orbit, Mean Anomaly propagates linearly
        # n_sc = sqrt(mu / a^3)
        n_sc = np.sqrt(mu_sun / sc_parking_elements['a']**3)
        # We need to know where the S/C is at t_burn. 
        # If t_burn is relative to detection, we need S/C state at detection.
        # Simplification: User provides True Anomaly at t_burn directly.
        nu_sc_burn = sim_params['nu_sc_at_burn']

        sc_burn_elements = sc_parking_elements.copy()
        sc_burn_elements['nu'] = nu_sc_burn

        r_sc, v_sc_initial = self._OEtoRV(sc_burn_elements)
        r_sc = r_sc.to(u.km).value
        v_sc_initial = v_sc_initial.to(u.km / u.s).value

        # 4. Apply Chemical Burn (Impulse)
        # The user asked for an "Inclination matching burn... and propel".
        # This is an optimization variable. For this walkthrough, we accept a Delta V vector.
        dv_chemical_vector = sim_params['dv_chemical_vector'] # numpy array [vx, vy, vz]
        
        v_sc_after_burn = v_sc_initial + dv_chemical_vector
        
        # Constraint Check: Chemical Delta V limit
        dv_chem_mag = np.linalg.norm(dv_chemical_vector)
        print(f"Chemical Burn magnitude: {dv_chem_mag:.3f} km/s")

        # 5. Propagate Electric Trajectory
        a_electric = sim_params['a_electric']
        initial_state = np.concatenate((r_sc, v_sc_after_burn))

        # Integrate for a max duration
        max_duration = sim_params['max_flight_time']
        sol = self.propagate_intercept((t_burn, t_burn + max_duration), initial_state, a_electric, None)

        # 6. Analyze Trajectory for Intercept
        min_dist = float('inf')
        intercept_idx = -1
        rel_vel_at_intercept = 0
        intercept_data = {}

        for i, t in enumerate(sol.t):
            # Get ISO state at this timestep
            r_iso, v_iso = self.get_iso_state(t, iso_elements)
            r_iso = r_iso.to(u.km).value
            v_iso = v_iso.to(u.km / u.s).value
            
            # Define Intercept Target: 1000 km in FRONT of ISO
            # Direction is v_iso / |v_iso|
            v_iso_unit = v_iso / np.linalg.norm(v_iso)
            r_target = r_iso + (1000.0 * v_iso_unit)
            
            # S/C state
            r_sc_curr = sol.y[:3, i]
            v_sc_curr = sol.y[3:, i]
            
            dist = np.linalg.norm(r_sc_curr - r_target)
            
            if dist < min_dist:
                min_dist = dist
                intercept_idx = i
                
                # Check constraints if this were the intercept
                dv_electric = a_electric.to(u.km / u.s**2).value * (t - t_burn) # Total delta V used by electric
                total_dv = dv_chem_mag + dv_electric
                dist_from_earth = np.linalg.norm(r_sc_curr - self.get_earth_pos(t)) # Placeholder for Earth pos
                
                intercept_data = {
                    'time': t,
                    'distance_miss': dist,
                    'r_sc': r_sc_curr,
                    'v_rel': np.linalg.norm(v_sc_curr - v_iso),
                    'total_dv': total_dv,
                    'pass_constraints': total_dv < 15.0 # Add other constraints here
                }

        return intercept_data

    def get_earth_pos(self, t):

        '''
        Calculate Earth's position at time t (seconds from J2000).
        '''
        # Constants
        mu_sun = (c.G * c.M_sun).to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]
        r = (1 * u.au).to(u.km).value

        # Simplified Earth position (Circular approx for demo)
        # Real implementation would use an ephemeris (e.g., spiceypy)
        n_earth = np.sqrt(mu_sun / r**3)
        theta = n_earth * t

        return np.array([r * np.cos(theta), r * np.sin(theta), 0])
    
def RVtoOE(r_barycentric, v_barycentric, mu):
    '''
    Convert barycentric state vectors to barycentric orbital elements.

    Parameters
    ----------
    r_barycentric : np.ndarray
        Position vector in barycentric frame [km]
    v_barycentric : np.ndarray
        Velocity vector in barycentric frame [km/s]
    mu : Quantity with km^3/s^2 unit
        Gravitational parameter of the central body [km^3/s^2]

    Returns
    -------
    orbital_elements : dict
        Dictionary containing heliocentric orbital elements:
        - a: Semi-major axis (astropy Quantity with length unit)
        - e: Eccentricity (float)
        - i: Inclination (astropy Quantity with angle unit)
        - Omega: Longitude of ascending node (astropy Quantity with angle unit)
        - omega: Argument of perihelion (astropy Quantity with angle unit)
        - M: Mean anomaly (astropy Quantity with angle unit)
        - epoch: Epoch of the elements (astropy Time object)
    '''
    r_barycentric = r_barycentric.to(u.km).value
    v_barycentric = v_barycentric.to(u.km / u.s).value

    # Defining constants
    mu_body = mu.to(u.km**3 / u.s**2).value # Gravitational parameter of the Sun [km^3/s^2]

    # Now, we need to convert the barycentric state vector back to barycentric orbital elements
    r_mag_barycentric = np.linalg.norm(r_barycentric)
    v_mag_barycentric = np.linalg.norm(v_barycentric)

    # Specific angular momentum vector
    h_barycentric = np.cross(r_barycentric, v_barycentric)
    h_mag_barycentric = np.linalg.norm(h_barycentric)

    # Calculating semimajor axis (a) [km]
    a = 1 / (2 / np.linalg.norm(r_barycentric) - np.linalg.norm(v_barycentric)**2 / mu_body)

    # Calculating eccentricity (e)
    e_vector = (1/mu_body) * (np.cross(v_barycentric, h_barycentric) - mu_body*r_barycentric/r_mag_barycentric)
    e = np.linalg.norm(e_vector)

    # Calculating orbital frame basis vectors
    i_e = e_vector / e  # Unit vector in direction of eccentricity
    i_h = h_barycentric / h_mag_barycentric  # Unit vector in direction of angular momentum
    i_y = np.cross(i_h, i_e)  # Unit vector perpendicular to both

    # Constructing DCM from inertial to orbital frame
    R_OI_barycentric = np.vstack((i_e, i_y, i_h))

    # Calculating inclination (i) [rad]
    i = np.arccos(R_OI_barycentric[2, 2])

    # Calculating longitude of ascending node (Omega) [rad]
    Omega = np.arctan2(i_h[0]/np.sin(i), -i_h[1]/np.sin(i))

    if Omega < 0:
        Omega += 2 * np.pi

    # Calculating argument of perihelion (omega) [rad]
    omega = np.arctan2(i_e[2]/np.sin(i), i_y[2]/np.sin(i))

    if omega < 0:
        omega += 2 * np.pi

    # Calculating true anomaly (nu) [rad]
    r_orbital = R_OI_barycentric @ r_barycentric
    nu = np.arctan2(r_orbital[1], r_orbital[0])

    # # Calculating hyperbolic eccentric anomaly (F) [rad]
    # F = 2 * np.arctanh( np.tan(nu / 2) * np.sqrt((e - 1) / (e + 1)) )

    # # Calculating hyperbolic mean anomaly (M) [rad]
    # M = e * np.sinh(F) - F

    # Inserting values into dictionary and converting to appropriate units
    orbital_elements = {
        'a': (a * u.km).to(u.au),
        'e': e,
        'i': (i * u.rad).to(u.deg),
        'Omega': (Omega * u.rad).to(u.deg),
        'omega': (omega * u.rad).to(u.deg),
        'nu': (nu * u.rad).to(u.deg),
    }

    return orbital_elements