import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'

num = 1000
t = np.linspace(0, 2*np.pi, num)
x = np.cos(t)
y = np.sin(t)
z = t / np.pi

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(2, 2, 2)
ax2 = fig.add_subplot(2, 2, 4)

line, = ax.plot(x, y, z, color='blue', label='Trajectory')
point, = ax.plot([], [], [], color='orange', marker='*', markersize=10, label='Sun', linestyle='')

ax.set_aspect('equal')

ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])

ax.set_xlabel('X [au]')
ax.set_ylabel('Y [au]')
ax.set_zlabel('Z [au]')

def init():
    point.set_data([], [])
    point.set_3d_properties([])

    return point

def update(frame):
    point.set_data([x[frame]], [y[frame]])
    point.set_3d_properties([z[frame]])

    return point

ani = animation.FuncAnimation(fig, update, frames=num, init_func=init, blit=False, interval=1)

plt.show()