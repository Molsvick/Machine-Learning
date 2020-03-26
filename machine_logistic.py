import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
#%matplotlib inline
import matplotlib.pyplot as plt

df = pd.read_csv("ChurnData.csv")
df.head()

df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
df['churn'] = df['churn'].astype('int')
print(df.head())

#Obtener número de columnas
n = df.shape[1]
print("Number of columns is:", n)

#Definimos X e Y para nuestro set datos
X = np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

y = np.asarray(df['churn'])
y [0:5]

#Normalizamos set de datos
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


