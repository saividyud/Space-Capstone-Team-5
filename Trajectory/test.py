import astropy.units as u
import numpy as np

t1 = 10 * u.day
v1 = 5 * u.km / u.day

print(f"Distance: {v1 * t1:.2f}")