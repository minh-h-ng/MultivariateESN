import pandas as pd
import numpy as np

filePath = ['/home/minh/PycharmProjects/MultivariateESN/data_generator/rainfall_temp/rainfall.csv',
            '/home/minh/PycharmProjects/MultivariateESN/data_generator/rainfall_temp/temperature.csv']
writePath = ['/home/minh/PycharmProjects/MultivariateESN/data_backup/Rainfall',
             '/home/minh/PycharmProjects/MultivariateESN/data_backup/Temperature']

for i in range(len(filePath)):
    df = pd.read_csv(filePath[i])
    data = np.array(df)
    #print('loadSP:',data)

    closeData = []
    for j in range(len(data)):
        closeData.append(data[j][0])

    with open(writePath[i],'w') as f:
        for j in range(len(closeData)-1):
            f.write(str(float(closeData[j]))+ ',' + str(float(closeData[j+1])) + '\n')