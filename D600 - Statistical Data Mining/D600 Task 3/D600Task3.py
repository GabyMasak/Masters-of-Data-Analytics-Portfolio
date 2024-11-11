#Gaby Masak
#D600 Task 3

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Import CSV file
realEstateData = pd.read_csv(r'C:\Users\gabri\OneDrive\Documents\Education\WGU\MSDA\D600 - Statistical Data Mining\D600 Task 3\D600 Task 3 Dataset 1 Housing Information.csv')

# Select the relevant columns for prediction
pricePrediction = realEstateData[['Price', 'SquareFootage', 'NumBathrooms', 'BackyardSpace', 'CrimeRate', 'SchoolRating', 'AgeOfHome', 'DistanceToCityCenter', 'EmploymentRate', 'RenovationQuality', 'LocalAmenities', 'TransportAccess']].copy()

# Standardize the variables
scaler = StandardScaler()
predictionIndependent = pricePrediction.drop('Price', axis=1)
pricePrediction_scaled = scaler.fit_transform(predictionIndependent)

# Convert the scaled data back to a DataFrame
pricePrediction_scaled_df = pd.DataFrame(pricePrediction_scaled, columns=predictionIndependent.columns)

# Display the first few rows of the standardized data and export to CSV
print(pricePrediction_scaled_df.head())
pricePrediction_scaled_df.to_csv(r'C:\Users\gabri\PycharmProjects\pythonProject\WGU MSDA\D600Task3Standardized.csv', index=False)

# Perform PCA
pca = PCA()
principalComponents = pca.fit_transform(pricePrediction_scaled)

# Determine the matrix of all principal components
principalComponents_df = pd.DataFrame(principalComponents, columns=[f'PC{i+1}' for i in range(principalComponents.shape[1])])

# Plot the Eigen values to use the Kaiser rule
eigenvalues = pca.explained_variance_
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
plt.axhline(y=1, color='r', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()

# Identify the total number of principal components to retain using the Kaiser rule
n_components_kaiser = sum(eigenvalues > 1)
print(f'Total number of principal components to retain using the Kaiser rule: {n_components_kaiser}')

# Identify the variance of each of the retained principal components
explained_variance = pca.explained_variance_ratio_[:n_components_kaiser]
for i, variance in enumerate(explained_variance, start=1):
    print(f'Variance of PC{i}: {variance:.4f}')

# Summarize the results of your PCA
print("\nSummary of PCA Results:")
print(f"Total variance explained by the first {n_components_kaiser} principal components: {sum(explained_variance):.4f}")
print(f"Principal components retained: {n_components_kaiser}")
print(f"Explained variance by each retained component: {explained_variance}")

# Print the matrix of all principal components
print("\nMatrix of all principal components:")
print(principalComponents_df)

# Retain the principal components identified by the Kaiser rule
principalComponents_retained = principalComponents_df.iloc[:, :n_components_kaiser]

# Split the dataset into training and testing sets using the retained principal components
X_train_pca, X_test_pca, y_train, y_test = train_test_split(principalComponents_retained, pricePrediction['Price'], test_size=0.2, random_state=42)

# Convert to DataFrames
train_set = pd.concat([X_train_pca, y_train], axis=1)
test_set = pd.concat([X_test_pca, y_test], axis=1)

# Export to csv files
train_set.to_csv(r'C:\Users\gabri\PycharmProjects\pythonProject\WGU MSDA\D600Task3TrainingSet.csv', index=False)
test_set.to_csv(r'C:\Users\gabri\PycharmProjects\pythonProject\WGU MSDA\D600Task3TestSet.csv', index=False)

# Add a constant to the model (intercept)
X_train_pca = sm.add_constant(X_train_pca)
X_test_pca = sm.add_constant(X_test_pca)

# Fit the optimized model
optimized_model = sm.OLS(y_train, X_train_pca).fit()

# Print the summary of the optimized model
print(optimized_model.summary())

# Print the regression coefficients and intercept
print("\nRegression Coefficients:")
print(optimized_model.params)

# Evaluate the model on the training set
y_train_pred = optimized_model.predict(X_train_pca)
mse_train = mean_squared_error(y_train, y_train_pred)
print(f'Mean Squared Error on Training Set: {mse_train:.4f}')

# Evaluate the model on the test set
y_test_pred = optimized_model.predict(X_test_pca)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'Mean Squared Error on Test Set: {mse_test:.4f}')
