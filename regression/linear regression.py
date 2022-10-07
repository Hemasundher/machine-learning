import numpy as n
import matplotlib.pyplot as m
import pandas as p
from sklearn import linear_model

df = p.read_csv("FuelConsumptionCo2.csv")
# print(df.head(5))
# m.xlabel('engine size')
# m.ylabel('co2')
#
# m.scatter(df['ENGINESIZE'],df['CO2EMISSIONS'],color="red",marker="+")
# m.show()

# print(type(df['CO2EMISSIONS']))

mask = n.random.rand(len(df)) < 0.8

train = df[mask]
test = df[~mask]

# print(len(df))
# print(len(train),len(test))


reg = linear_model.LinearRegression()
reg.fit(train[['ENGINESIZE']], train[['CO2EMISSIONS']])
print(reg.coef_)
print(reg.intercept_)

pre_y = reg.predict(test[['ENGINESIZE']])

print("mean absolute error : ", n.mean(n.absolute( pre_y-test[['CO2EMISSIONS']] )))
print(" ")
print(reg.score(test[['ENGINESIZE']], test[['CO2EMISSIONS']]))
