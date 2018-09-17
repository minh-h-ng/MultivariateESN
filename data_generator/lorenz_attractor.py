import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

writePath = '/home/minh/PycharmProjects/MultivariateESN/data/Lorenz_noise'

def lorenz(x, y, z, s=10.0, r=28.0, b=8/3):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

dt = 0.01
stepCnt = 10000

# Need one more for the initial values
xs = np.empty((stepCnt + 1,))
ys = np.empty((stepCnt + 1,))
zs = np.empty((stepCnt + 1,))

# Setting initial values
xs[0], ys[0], zs[0] = (1.0, 1.0, 1.0)

# Stepping through "time".
for i in range(stepCnt):
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

trainSize = int(4000 * 0.5)
signalStd = np.std(xs[:trainSize])
noiseStd = 0.2 * signalStd

# add noise
noise = np.random.normal(0,noiseStd,(trainSize))
print('noise:',noise)
for i in range(len(noise)):
    xs[i] = xs[i] + noise[i]


with open(writePath,'w') as f:
    for i in range(4000):
        f.write(str(xs[i]) + ',' + str(ys[i]) + ',' + str(zs[i]) + ',' +
                str(xs[i+1]) + '\n')

"""trainSize = 2000-400
outliers = random.sample(range(trainSize),15)
print('outliers:',outliers)
for i in outliers:
    xs[i] = 50 * xs[i]"""

"""fig = plt.figure()
axes = plt.gca()
axes.set_ylim([-20,20])
plt.plot(xs[2000:2500])

plt.show()"""

"""fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()"""