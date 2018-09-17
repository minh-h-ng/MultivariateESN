from os import listdir
from sklearn.metrics import mean_squared_error
import numpy as np

basePath = '/home/minh/PycharmProjects/MultivariateESN/predictions/'

def RMSE(reals,predictions):
    return np.sqrt(mean_squared_error(reals,predictions))

def MSE(reals,predictions):
    return mean_squared_error(reals,predictions)

def MAE(reals,predictions):
    mae = 0
    for i in range(len(reals)):
        mae += abs(predictions[i]-reals[i])
    return mae/len(reals)

def generateMSE(fileName):
    filePath = basePath + fileName
    data = np.genfromtxt(filePath,delimiter=',')
    data = np.transpose(data)
    reals = data[0]
    MSEList = []
    MAEList = []
    RMSEList = []
    for i in range(1,len(data)):
        MSEList.append(MSE(reals,data[i]))
        MAEList.append(MAE(reals,data[i]))
        RMSEList.append(RMSE(reals,data[i]))
    print('model:',fileName,'MAE:',np.mean(MAEList),'MSE:',np.mean(MSEList),'RMSE:',np.mean(RMSEList))

for fileName in listdir(basePath):
    nameSplit = fileName.split('_')
    if nameSplit[0]=='Hongik':
        mseRes = generateMSE(fileName)