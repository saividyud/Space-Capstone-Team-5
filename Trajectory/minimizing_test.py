import numpy as np
from scipy.optimize import minimize

# 1. The Objective Function
# Minimize f(x) = x^2 + y^2
def objective(x):
    return x[0]**2 + x[1]**2

# 2. The Constraint
# We want: x + y >= 1
# SciPy requires constraints to be written as: fun(x) >= 0
# Therefore, we rearrange to: (x + y) - 1 >= 0
def constraint_func(x):
    return x[0] + x[1] - 1

# Define the dictionary
# 'type' can be 'eq' (equality) or 'ineq' (inequality)
cons = {'type': 'ineq', 'fun': constraint_func}

# 3. Execution
x0 = [2.0, 2.0] # Initial guess
solution = minimize(objective, x0, method='SLSQP', constraints=cons)

print(f"Success: {solution.success}")
print(f"Solution x: {solution.x}") 
print(f"Objective Value: {solution.fun}")