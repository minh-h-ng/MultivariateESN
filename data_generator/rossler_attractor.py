import numpy as np
import matplotlib.pyplot as plt
import os
import random

writePath = '/home/minh/PycharmProjects/MultivariateESN/data/Rossler_noise'

def rossler(x, y, z, s=0.2, r=5.7, b=0.2):
    x_dot = -y - z
    y_dot = x + s*y
    z_dot = b + z*(x-r)
    return x_dot, y_dot, z_dot


dt = 0.05
stepCnt = 10000

# Need one more for the initial values
xs = np.empty((stepCnt + 1))
ys = np.empty((stepCnt + 1))
zs = np.empty((stepCnt + 1))

# Setting initial values
xs[0], ys[0], zs[0] = (-1, 0, 3)

# Stepping through "time".
for i in range(stepCnt):
    # Derivatives of the X, Y, Z state
    x_dot, y_dot, z_dot = rossler(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

if os.path.exists(writePath):
    os.remove(writePath)
os.mknod(writePath)

trainSize = int(4000*0.5)
signalStd = np.std(xs[:trainSize])
noiseStd = 0.2 * signalStd

""""# add 15 outliers
outliers = random.sample(range(trainSize),15)
print('outliers:',outliers)
for i in outliers:
    xs[i] = 50 * xs[i]"""

# add noise
noise = np.random.normal(0,noiseStd,(trainSize))
print('noise:',noise)
for i in range(len(noise)):
    xs[i] = xs[i] + noise[i]

with open(writePath,'w') as f:
    for i in range(4000):
        f.write(str(xs[i]) + ',' + str(ys[i]) + ',' + str(zs[i]) + ',' +
                str(xs[i+1]) + '\n')

"""fig = plt.figure()
axes = plt.gca()
axes.set_ylim([-20,20])
plt.plot(xs[2000:4000])

plt.show()"""