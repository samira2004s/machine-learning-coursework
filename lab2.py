import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split

# Load dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(url)

# Drop unused columns
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB'], axis=1)

# Keep only numeric columns (remove MAKE, MODEL, FUELTYPE which are strings)
df_numeric = df.select_dtypes(include=[np.number])

# Scatter matrix for numeric columns
axes = pd.plotting.scatter_matrix(df_numeric, alpha=0.2, figsize=(8, 8))
for ax in axes.ravel():
    ax.set_xlabel(ax.get_xlabel(), rotation=90)
    ax.set_ylabel(ax.get_ylabel(), rotation=0)
    ax.yaxis.label.set_ha('right')
plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()

# Prepare features (first two numeric columns) and target (third numeric column)
X = df_numeric.iloc[:, [0, 1]].to_numpy()
y = df_numeric.iloc[:, 2].to_numpy()

# Standardize features
std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

print("Standardized feature stats:\n", pd.DataFrame(X_std).describe().round(2))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

# Train multiple regression model
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Output standardized results
coef_ = regressor.coef_
intercept_ = regressor.intercept_
print('\n--- Standardized Model ---')
print('Coefficients:', coef_)
print('Intercept:', intercept_)
print('R² score (train):', regressor.score(X_train, y_train))
print('R² score (test):', regressor.score(X_test, y_test))

# Convert coefficients back to original scale
means_ = std_scaler.mean_
std_devs_ = std_scaler.scale_  # correct standard deviations
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

# Output original-scale results
print('\n--- Original Scale Model ---')
print('Coefficients:', coef_original)
print('Intercept:', intercept_original)

from mpl_toolkits.mplot3d import Axes3D

# Use standardized X_test for prediction
X1 = X_test[:, 0]
X2 = X_test[:, 1]

# Create meshgrid in standardized space
x1_surf, x2_surf = np.meshgrid(
    np.linspace(X1.min(), X1.max(), 100),
    np.linspace(X2.min(), X2.max(), 100)
)

# Predict values on the meshgrid
y_surf = regressor.intercept_ + regressor.coef_[0] * x1_surf + regressor.coef_[1] * x2_surf

# Predictions for actual test data
y_pred = regressor.predict(X_test)

# Determine above/below plane
above_plane = y_test >= y_pred
below_plane = y_test < y_pred

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Data points
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],
           color='blue', label="Above Plane", s=70, alpha=0.7, edgecolor='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],
           color='red', label="Below Plane", s=50, alpha=0.3, edgecolor='k')

# Regression plane
ax.plot_surface(x1_surf, x2_surf, y_surf, color='gray', alpha=0.3)

# Labels
ax.set_xlabel('ENGINESIZE (standardized)', fontsize=14)
ax.set_ylabel('FUELCONSUMPTION (standardized)', fontsize=14)
ax.set_zlabel('CO2 Emissions', fontsize=14)
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize=16)
ax.view_init(elev=10)
ax.legend()
plt.tight_layout()
plt.show()

