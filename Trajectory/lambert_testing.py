from poliastro import iod
import Functions as Funcs
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as c



# # Testing Lambert Solver
# r0 = np.array([-0.3730, -1.4581, 0.5976]) * 10**7 * u.m
# r1 = np.array([1.850, -2.1920, 0.0431]) * 10**7 * u.m
# tof = 1*u.hr + 38*u.min + 46*u.s

# v0, v1 = iod.lambert(k=c.GM_earth, r0=r0, r=r1, tof=tof).__next__()

# # print(f"Initial Velocity (v0): {v0}")
# # print(f"Final Velocity (v1): {v1}")
# print(v0)  # Get the first solution
# print(v1)  # Get the final solution

# print(Funcs.RVtoOE(r0, v0, mu=c.GM_earth))  # Semi-major axis in km