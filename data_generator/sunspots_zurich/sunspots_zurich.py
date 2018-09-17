import pandas as pd
import numpy as np

filePath = '/home/minh/PycharmProjects/MultivariateESN/data_generator/sunspots_zurich/sunspots_zurich.csv'
writePath = '/home/minh/PycharmProjects/MultivariateESN/data_backup/SunSpotsZu'

df = pd.read_csv(filePath)
data = np.array(df)
#print('loadSP:',data)

closeData = []
for j in range(len(data)):
    closeData.append(data[j][1])

with open(writePath,'w') as f:
    for j in range(len(closeData)-1):
        f.write(str(float(closeData[j]))+ ',' + str(float(closeData[j+1])) + '\n')