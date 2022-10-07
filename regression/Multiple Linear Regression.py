import numpy as n
import matplotlib.pyplot as m
import pandas as p
from sklearn import linear_model

df = p.read_csv("FuelConsumptionCo2.csv")


mask = n.random.rand(len(df)) < 0.8

train = df[mask]
test = df[~mask]

reg=linear_model.LinearRegression()
reg.fit(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']],train[['CO2EMISSIONS']])

print(reg.coef_,reg.intercept_)


