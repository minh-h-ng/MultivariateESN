import csv

"""filePath = './results/running_time_1.csv'
writePath = './results/running_time_1_final.csv'

models = ['bayelinear_identity_','bayeridge_identity_','linsvr_identity_','nusvr_identity_','ridge_identity_']

for model in models:
    for i in range(100, 1001, 100):
        runTime = 0
        count = 0
        curModel = model + str(i)
        with open(filePath, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if line[0]==curModel:
                    runTime += float(line[-1])
                    count+=1
        avgTime = [curModel, runTime / count]
        with open(writePath, 'a') as g:
            writer = csv.writer(g)
            writer.writerow(avgTime)

filePath = './results/running_time_2.csv'
writePath = './results/running_time_2_final.csv'
models = ['bayeridge_pca_','bayeridge_kpca_','bayeridge_ica_']

for model in models:
    for i in range(100, 1001, 100):
        runTime = 0
        count = 0
        curModel = model + str(i)
        with open(filePath, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if line[0]==curModel:
                    runTime += float(line[-1])
                    count+=1
        avgTime = [curModel, runTime / count]
        with open(writePath, 'a') as g:
            writer = csv.writer(g)
            writer.writerow(avgTime)"""

filePath = './results/running_time_3.csv'
writePath = './results/running_time_3_final.csv'
models = ['bayelinear_identity_','bayeridge_identity_','linsvr_identity_','nusvr_identity_',
          'ridge_identity_','bayeridge_pca_','bayeridge_kpca_','bayeridge_ica_']

for model in models:
    for i in range(1100, 1501, 100):
        runTime = 0
        count = 0
        curModel = model + str(i)
        with open(filePath, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if line[0]==curModel:
                    runTime += float(line[-1])
                    count+=1
        avgTime = [curModel, runTime / count]
        with open(writePath, 'a') as g:
            writer = csv.writer(g)
            writer.writerow(avgTime)