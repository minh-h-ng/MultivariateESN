def HenonMap(a,b,x,y):
    return y+1.0-a*x*x, b*x

iterates = 10000

a = 1.4
b = 0.3

xtemp = 1.0
ytemp = 1.0

x = [xtemp]
y = [ytemp]

for n in range(iterates):
    xtemp, ytemp = HenonMap(a,b,xtemp,ytemp)
    x.append(xtemp)
    y.append(ytemp)

writePath = '/home/minh/PycharmProjects/MultivariateESN/data_backup/HenonMap'
no_data = 4000

with open(writePath, 'w') as f:
    for i in range(100,no_data+100):
        f.write(str(x[i]) + ',' + str(y[i]) + ',' + str(y[i+1]) + '\n')