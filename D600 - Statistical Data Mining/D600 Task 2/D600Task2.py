#Gaby Masak
#D600 Task 2

import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Import CSV file
realEstateData = pd.read_csv(r'C:\Users\gabri\OneDrive\Documents\Education\WGU\MSDA\D600 - Statistical Data Mining\D600 Task 1\D600 Task 1 Dataset 1 Housing Information.csv')
#print(realEstateData)

# Convert 'Garage' and 'Fireplace' columns from "Yes"/"No" to 1/0
realEstateData['Garage'] = realEstateData['Garage'].map({'Yes': 1, 'No': 0})
realEstateData['Fireplace'] = realEstateData['Fireplace'].map({'Yes': 1, 'No': 0})

# Separate into training and test sets
pricePrediction = realEstateData[['IsLuxury', 'Fireplace', 'Garage', 'NumBedrooms', 'RenovationQuality']].copy()
#print(pricePrediction)

# Define your features (X) and target (y)
X = pricePrediction.drop('IsLuxury', axis=1)
y = pricePrediction['IsLuxury']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("Training set size:", X_train.shape)
#print("Testing set size:", X_test.shape)

# Convert to DataFrames
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# Export to csv files
train_set.to_csv(r'C:\Users\gabri\PycharmProjects\pythonProject\WGU MSDA\D600TrainingSet2.csv', index=False)
test_set.to_csv(r'C:\Users\gabri\PycharmProjects\pythonProject\WGU MSDA\D600TestSet2.csv', index=False)

# Add a constant to the model (intercept)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Forward stepwise selection
def forward_selection(X, y):
    initial_features = X.columns.tolist()
    best_features = []
    while initial_features:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.Logit(y, sm.add_constant(X[best_features + [new_column]])).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < 0.05:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

selected_features = forward_selection(X_train, y_train)
print(f'Selected features: {selected_features}')

# Fit the optimized model
optimized_model = sm.Logit(y_train, sm.add_constant(X_train[selected_features])).fit()

# Extract model parameters
summary = optimized_model.summary2()
print(summary)

# Extract specific model parameters
aic = optimized_model.aic
bic = optimized_model.bic
pseudo_r2 = optimized_model.prsquared
coefficients = optimized_model.params
p_values = optimized_model.pvalues

print(f"AIC: {aic}")
print(f"BIC: {bic}")
print(f"Pseudo R2: {pseudo_r2}")
print(f"Coefficients: {coefficients}")
print(f"P-values: {p_values}")


# Predict on the training set
y_train_pred = optimized_model.predict(sm.add_constant(X_train[selected_features]))
y_train_pred_class = (y_train_pred >= 0.5).astype(int)
mse_train = mean_squared_error(y_train, y_train_pred_class)
accuracy_train = accuracy_score(y_train, y_train_pred_class)
conf_matrix_train = confusion_matrix(y_train, y_train_pred_class)

print(f"Training MSE: {mse_train}")
print(f"Training Accuracy: {accuracy_train}")
print(f"Training Confusion Matrix:\n{conf_matrix_train}")


# Predict on the training set and calculate residuals
y_train_pred = optimized_model.predict(sm.add_constant(X_train[selected_features]))
residuals = y_train - y_train_pred

# # Create a time variable (assuming your data is ordered by time)
# time = np.arange(len(y_train))
#
# # Plot residuals over time
# plt.figure(figsize=(10, 6))
# plt.plot(time, residuals, marker='o', linestyle='-', color='b')
# plt.xlabel('Time')
# plt.ylabel('Residuals')
# plt.title('Residuals Over Time')
# plt.axhline(y=0, color='r', linestyle='--')
# plt.show()


# Predict on the test set
y_test_pred = optimized_model.predict(sm.add_constant(X_test[selected_features]))
y_test_pred_class = (y_test_pred >= 0.5).astype(int)
mse_test = mean_squared_error(y_test, y_test_pred_class)
accuracy_test = accuracy_score(y_test, y_test_pred_class)
conf_matrix_test = confusion_matrix(y_test, y_test_pred_class)

print(f"Test MSE: {mse_test}")
print(f"Test Accuracy: {accuracy_test}")
print(f"Test Confusion Matrix:\n{conf_matrix_test}")

