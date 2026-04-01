# K-Nearest Neighbors Classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df['custcat'].value_counts()
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

correlation_values = abs(df.corr()['custcat'].drop(['custcat']).sort_values(ascending=False))

# Separate the Input and Target Features
X = df.drop('custcat', axis=1)
y = df['custcat']

# Normalize Data
X_norm = StandardScaler().fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# KNN Classification
Ks =10
acc= np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1, Ks+1):
    knn_model = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = knn_model.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# Plot the model accuracy for a different number of neighbors.
plt.plot(range(1, Ks+1), acc, 'g')
plt.fill_between(range(1, Ks+1), acc - std_acc, acc + std_acc, alpha=0.1)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 