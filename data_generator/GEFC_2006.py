import pandas as pd

basePath = '/home/minh/PycharmProjects/MultivariateESN/data_generator/GEFC2012/'
loadFile = 'Load_history.csv'
benchmarkFile = 'Benchmark.csv'
fullFile = 'Full_Load.csv'
sumFile = 'Sum_Load.csv'
tempFile = 'temperature_history.csv'
avgtempFile = 'Average_Temperature.csv'
sum2006File = 'Sum_Load_2006.csv'
avgtemp2006File = 'Average_Temperature_2006.csv'

# load load_history file and drop NaN rows
loadDF = pd.read_csv(basePath+loadFile)
print('load DF size:',loadDF.shape)
loadDF = loadDF[loadDF['h1'].notnull()]
print('load DF size after dropping NaN:',loadDF.shape)

# load benchmark file and drop row with zone_id 21
benchmarkDF = pd.read_csv(basePath+benchmarkFile)
print('benchmarkDF size:',benchmarkDF.shape)
benchmarkDF = benchmarkDF[benchmarkDF['zone_id']!=21]
print('benchmarkDF size after:',benchmarkDF.shape)

# concatenate 2 dataframes and sort
fullDF = pd.concat([loadDF,benchmarkDF],ignore_index=True)
print('fullDF size:',fullDF.shape)
fullDF = fullDF.sort_values(['year','month','day','zone_id'],ascending=[True,True,True,True])
print('fullDF size:',fullDF.shape)

# print full dataframe to csv
fullDF.to_csv(basePath+fullFile,columns=['zone_id','year','month','day','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10',
                                    'h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24'])


# remove commas, sum, and print to csv
fullDF = fullDF.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',',''),errors='coerce'))
sumDF = fullDF.groupby(['year','month','day'])['h1','h2','h3','h4','h5','h6','h7','h8','h9','h10',
                                    'h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24'].sum().reset_index()
sumDF.to_csv(basePath + sumFile)

# take sumDF of only year 2006, and print to csv
sumDF2006 = sumDF[sumDF['year']==2006].reset_index(drop=True)
sumDF2006.to_csv(basePath + sum2006File)

# load temperature_history
temperatureDF = pd.read_csv(basePath + tempFile)
#temperatureDF = temperatureDF.apply(pd.to_numeric)

# average, and print to csv
avgtempDF = temperatureDF.groupby(['year','month','day'])['h1','h2','h3','h4','h5','h6','h7','h8','h9','h10',
                                    'h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23','h24'].mean().reset_index()
avgtempDF.to_csv(basePath + avgtempFile)

# take avetempDF of only year 2006, and print to csv
avgtempDF2006 = avgtempDF[avgtempDF['year']==2006].reset_index(drop=True)
avgtempDF2006.to_csv(basePath + avgtemp2006File)
