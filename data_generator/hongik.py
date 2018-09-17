import csv
import numpy as np

writePath = '/home/minh/PycharmProjects/MultivariateESN/data/Hongik'

data = np.genfromtxt('hongik.csv')

with open(writePath,'w') as f:
    for i in range(len(data)-1):
        f.write(str(data[i]) + ',' + str(data[i+1]) + '\n')