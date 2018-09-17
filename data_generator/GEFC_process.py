import csv

basePath = '/home/minh/PycharmProjects/MultivariateESN/data_generator/GEFC2012/'
sum2006File = 'Sum_Load_2006.csv'
avgtemp2006File = 'Average_Temperature_2006.csv'

writeBasePath = '/home/minh/PycharmProjects/MultivariateESN/data_backup/'
sumOut = 'GEFC'
sumtempOut = 'GEFC_temp'

sum = []
count = 0
with open(basePath + sum2006File,'r') as f:
    reader = csv.reader(f)
    for line in reader:
        if count==0:
            pass
        else:
            line = line[4:]
            sum += line
        count+=1

for i in range(len(sum)):
    sum[i] = float(sum[i])/50000

temp = []
count = 0
with open(basePath + avgtemp2006File,'r') as f:
    reader = csv.reader(f)
    for line in reader:
        if count==0:
            pass
        else:
            line = line[4:]
            temp += line
        count+=1

with open(writeBasePath+sumOut,'w') as f:
    for i in range(len(sum)-1):
        f.write(str(sum[i]) + ',' + str(sum[i+1]) + '\n')

with open(writeBasePath+sumtempOut,'w') as f:
    for i in range(len(temp)-1):
        f.write(str(sum[i]) + ',' + temp[i] + ',' + str(sum[i+1]) + '\n')