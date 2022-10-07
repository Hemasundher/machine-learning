import numpy as n
import matplotlib.pyplot as m
import pandas as p
from sklearn import linear_model

df = p.read_csv("FuelConsumptionCo2.csv")
mask = n.random.rand(len(df)) < 0.8

train = df[mask]
test = df[~mask]

#
# m.scatter(df['ENGINESIZE'],df['CO2EMISSIONS'])
# m.show()


train_x = n.asanyarray(train[['ENGINESIZE']])
train_y = train['CO2EMISSIONS']

from sklearn.preprocessing import PolynomialFeatures

ploy = PolynomialFeatures(degree=2)

train_x_poly = ploy.fit_transform(train_x)
reg = linear_model.LinearRegression()
train_y_ = reg.fit(train_x_poly, train_y)

m.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'])
# XX=n.arange(0,10,0.1)

print(reg.intercept_, "\n", reg.coef_, "\n", type(reg.coef_), type(reg.intercept_))
# yy = reg.intercept_[0]+ reg.coef_[0][1]*XX+ reg.coef_[0][2]*n.power(XX, 2)
# m.plot(XX, yy, '-r' )
# m.show()
XX = n.arange(0.0, 10.0, 0.1)
yy = reg.intercept_[0] + reg.coef_[0][1] * XX + reg.coef_[0][2] * n.power(XX, 2)
m.plot(XX, yy, '-r')
