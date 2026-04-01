import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)
df.sample(5)
df.describe()
cdf = df[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.sample(9)
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel('fuel consumption')
plt.ylabel('co2 emission')
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel('Engine size')
plt.ylabel('CO2 Emissions')
plt.xlim(0,27)
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='pink')
plt.xlabel('Cylinder')
plt.ylabel('CO2 Emissions')
plt.show()

X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(type(X_train), np.shape(X_train))

from sklearn import linear_model
regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)
print(regressor.coef_[0])
print(regressor.intercept_)

plt.scatter(X_train,y_train)
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_test_ = regressor.predict(X_test.reshape(-1, 1))
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))

plt.scatter(X_test, y_test, color='blue')  # Plot actual test data points
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')  # Plot regression line
plt.xlabel("Engine size")
plt.ylabel("CO2 Emissions")
plt.show()

X2 = df.FUELCONSUMPTION_COMB.to_numpy()
y2 = df.CO2EMISSIONS.to_numpy()
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
regr = linear_model.LinearRegression()
regr.fit(X2_train.reshape(-1, 1), y2_train)
y2_test_ = regr.predict(X2_test.reshape(-1, 1))
print("Mean squared error: %.2f" % mean_squared_error(y2_test, y2_test_))






