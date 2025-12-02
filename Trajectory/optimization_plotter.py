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
    't_burn': -700 * u.day,   # Burn executed at 700 days before perihelion
    'nu_burn': 140 * u.deg, # S/C at aphelion/perihelion etc
    'a_electric': 0 * u.mm / u.s**2,  # mm/s^2 (Low thrust)
    'max_flight_time': 3000 * u.day,
    
    # This vector is what you would iterate on to minimize miss distance
    'dv_chemical_vector': np.array([1, 0, -5.5]) * u.km / u.s # Defined in the spacecraft orbital frame
}

mission = Funcs.InterceptorMission(iso_elements=Funcs.characteristic_heliocentric_elements,
                                  sc_elements=sc_elements,
                                  sim_params=sim_params)

result, anim1, anim2 = mission.solve_intercept_optimization()

anim1.save('./Trajectory/optimization_trajectory_example.mp4', writer='ffmpeg', fps=15)
anim2.save('./Trajectory/optimization_history.mp4', writer='ffmpeg', fps=5)

print('Optimization Result:')
print(mission.sim_params['dv_chemical_vector'])

# mission.intercept_trajectory()
# anim = mission.plot_trajectory()

plt.show()