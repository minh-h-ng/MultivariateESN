import pandas as pd
import numpy as np

filePath = '/home/minh/PycharmProjects/MultivariateESN/data_generator/temp_canada_2017/temp_canada_2017.csv'
writePath = '/home/minh/PycharmProjects/MultivariateESN/data_backup/TempDenver'

df = pd.read_csv(filePath)
data = np.array(df)
#print('loadSP:',data)

closeData = []
for j in range(len(data)):
    closeData.append(data[j][10])

with open(writePath,'w') as f:
    for j in range(len(closeData)-1):
        f.write(str(float(closeData[j]))+ ',' + str(float(closeData[j+1])) + '\n')