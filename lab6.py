# Regression Trees

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import  mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Dataset Analysis
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)

correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))

# Dataset Preprocessing
# Target variable
y = raw_data['tip_amount'].values.astype('float32')

# Features
proc_data = raw_data.drop(['tip_amount', 'store_and_fwd_flag', 'improvement_surcharge', 'passenger_count'], axis=1)
# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(proc_data.values)

from sklearn.tree import DecisionTreeRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_reg = DecisionTreeRegressor(criterion='squared_error', max_depth = 4, random_state=32)
dt_reg.fit(X_train, y_train)
y_pred= dt_reg.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(dt_reg.score(X_test, y_test))

correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
print(abs(correlation_values).sort_values(ascending = True)[:4])





