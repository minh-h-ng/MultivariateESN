import pandas as pd
import numpy as np

filePath = 'SP500.csv'
writePath = '/home/minh/PycharmProjects/MultivariateESN/data_backup/SP500'

df = pd.read_csv(filePath)
data = np.array(df)
#print('loadSP:',data)

closeData = []
for i in range(len(data)):
    closeData.append(data[i][-2])

with open(writePath,'w') as f:
    for i in range(len(closeData)-1):
        f.write(str(float(closeData[i]))+ ',' + str(float(closeData[i+1])) + '\n')