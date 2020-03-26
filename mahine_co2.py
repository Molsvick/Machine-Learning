import inline as inline
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline
from sklearn import linear_model

df = pd.read_csv("FuelConsumptionCo2.csv")
df.hist()
plt.show()

cdf = df[['CYLINDERS', 'CO2EMISSIONS', 'ENGINESIZE']]
cdf.head(13)

viz = cdf[['CYLINDERS', 'CO2EMISSIONS', 'ENGINESIZE']]
viz.hist()
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.ENGINESIZE,  color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("ENGINE SIZE")
plt.show()

plt.scatter(cdf.CO2EMISSIONS, cdf.ENGINESIZE, color='red')
plt.xlabel("C02 Emi")
plt.ylabel("Engine Size")
plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()