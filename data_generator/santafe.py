import numpy as np

dataPath = '/home/minh/PycharmProjects/MultivariateESN/data_generator/SF_A.dat'
dataPath_cont = '/home/minh/PycharmProjects/MultivariateESN/data_generator/SF_Acont.dat'

writePath = '/home/minh/PycharmProjects/MultivariateESN/data/SantaFe'

data = np.loadtxt(dataPath,delimiter=',')
#print('data:',data)
data_cont = np.loadtxt(dataPath_cont,delimiter=',')

#only use the data_cont for now
with open(writePath,'w') as f:
    for i in range(len(data_cont)-1):
        f.write(str(float(data_cont[i]))+ ',' + str(float(data_cont[i+1])) + '\n')