import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

import warnings
warnings.filterwarnings('ignore')


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(url)
print(churn_df)

churn_df = churn_df[['tenure', 'age', 'address', 'ed', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')


X = np.asarray(churn_df[['tenure', 'age', 'address', 'ed']])
y = np.asarray(churn_df['churn'])

X_norm = StandardScaler().fit(X).transform(X)

# splitting the data set
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# Logistic Regression Classifier Modeling
LR = LogisticRegression().fit(X_train, y_train) #fitting = training

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

coefficients = pd.Series(LR.coef_[0],index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()

# Log Loss
print(log_loss(y_test, yhat_prob))