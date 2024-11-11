#Gaby Masak
#D600 Task 1

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

#Import CSV file
realEstateData = pd.read_csv(r'C:\Users\gabri\OneDrive\Documents\Education\WGU\MSDA\D600 - Statistical Data Mining\D600 Task 1\D600 Task 1 Dataset 1 Housing Information.csv')
#print(realEstateData)

#Descriptive Statistics of Variables
#Price
#count
#print(f'Count of Price: ',realEstateData['Price'].count())
#mean
#print(f'Mean of Price: ', realEstateData['Price'].mean())
#mode
#print(f'Mode of Price: ', realEstateData['Price'].mode())
#min/max
#min_price = realEstateData['Price'].min()
#print(f'Minimum Price: ', min_price)
#max_price = realEstateData['Price'].max()
#print(f'Maximum Price: ', max_price)
#range
#print(f'Range of Price: ', max_price-min_price)

#Square Footage
#count
#print(f'Count of Square Footage: ',realEstateData['SquareFootage'].count())
#mean
#print(f'Mean of Square Foortage: ', realEstateData['SquareFootage'].mean())
#mode
#print(f'Mode of Square Footage: ', realEstateData['SquareFootage'].mode())
#min/max
#min_sqft = realEstateData['SquareFootage'].min()
#print(f'Minimum Square Footage: ', min_sqft)
#max_sqft = realEstateData['SquareFootage'].max()
#print(f'Maximum Square Footage: ', max_sqft)
#range
#print(f'Range of Square Footage: ', max_sqft-min_sqft)

#Number of Bathrooms
#count
#print(f'Count of NumBathrooms: ',realEstateData['NumBathrooms'].count())
#mean
#print(f'Mean of NumBathrooms: ', realEstateData['NumBathrooms'].mean())
#mode
#print(f'Mode of NumBathrooms: ', realEstateData['NumBathrooms'].mode())
#min/max
#min_baths = realEstateData['NumBathrooms'].min()
#print(f'Minimum NumBathrooms: ', min_baths)
#max_baths = realEstateData['NumBathrooms'].max()
#print(f'Maximum NumBathrooms: ', max_baths)
#range
#print(f'Range of NumBathrooms: ', max_baths-min_baths)

#Number of Bedrooms
#count
#print(f'Count of NumBedrooms: ',realEstateData['NumBedrooms'].count())
#mean
#print(f'Mean of NumBedrooms: ', realEstateData['NumBedrooms'].mean())
#mode
#print(f'Mode of NumBedrooms: ', realEstateData['NumBedrooms'].mode())
#min/max
#min_beds = realEstateData['NumBedrooms'].min()
#print(f'Minimum NumBedrooms: ', min_beds)
#max_beds = realEstateData['NumBedrooms'].max()
#print(f'Maximum NumBedrooms: ', max_beds)
#range
#print(f'Range of NumBedrooms: ', max_beds-min_beds)

#BackyardSpace
#count
#print(f'Count of Backyard Space: ',realEstateData['BackyardSpace'].count())
#mean
#print(f'Mean of Backyard Space: ', realEstateData['BackyardSpace'].mean())
#mode
#print(f'Mode of Backyard Space: ', realEstateData['BackyardSpace'].mode())
#min/max
#min_space = realEstateData['BackyardSpace'].min()
#print(f'Backyard Space Min: ', min_space)
#max_space = realEstateData['BackyardSpace'].max()
#print(f'Backyard Space Max: ', max_space)
#range
#print(f'Backyard Space Range: ', max_space-min_space)

#Separate into training and test sets
#pricePrediction = realEstateData[['Price', 'SquareFootage', 'NumBathrooms', 'NumBedrooms', 'BackyardSpace']].copy()
#Drop BackyardSpace variable for optimized model
pricePrediction = realEstateData[['Price', 'SquareFootage', 'NumBathrooms', 'NumBedrooms']].copy()
#print(pricePrediction)
# Define your features (X) and target (y)
X = pricePrediction.drop('Price', axis=1)
y = pricePrediction['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print("Training set size:", X_train.shape)
#print("Testing set size:", X_test.shape)

# Convert to DataFrames
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

#Export to csv files
train_set.to_csv(r'C:\Users\gabri\PycharmProjects\pythonProject\WGU MSDA\D600TrainingSet.csv', index= False)
test_set.to_csv(r'C:\Users\gabri\PycharmProjects\pythonProject\WGU MSDA\D600TestSet.csv', index= False)

#Linear Regression Analysis
# Initialize the model
model = LinearRegression()

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
            model = sm.OLS(y, sm.add_constant(X[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < 0.05:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

#Show variables
selected_variables = forward_selection(X_train, y_train)
print(f'Selected features: {selected_variables}')


# Fit the model
model = sm.OLS(y_train, X_train).fit()

# Extract the model summary
summary = model.summary()
print(summary)

# Extract coefficients and intercept
intercept = model.params[0]
coefficients = model.params[1:]

# Print the regression equation
equation = f"Price = {intercept:.2f}"
for i, col in enumerate(X_train.columns[1:]):  # Skip the constant term
    equation += f" + ({coefficients[i]:.2f} * {col})"
print("Regression Equation:")
print(equation)


# Calculate MSE on the training set
y_train_pred = model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
print(f"Training MSE: {mse_train}")

# Calculate MSE on the test set
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Test MSE: {mse_test}")

#Testing Assumptions
# Linearity
plt.scatter(y_train, y_train_pred)
plt.xlabel('Observed values')
plt.ylabel('Predicted values')
plt.title('Observed vs. Predicted values')
plt.show()

# Independence of Residuals
dw_test = sm.stats.durbin_watson(model.resid)
print(f'Durbin-Watson test: {dw_test}')

# Homoscedasticity
plt.scatter(y_train_pred, model.resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted values')
plt.show()

# Normality of Residuals
sm.qqplot(model.resid, line='s')
plt.title('Q-Q plot')
plt.show()

shapiro_test = shapiro(model.resid)
print(f'Shapiro-Wilk test: {shapiro_test}')

# Multicollinearity
def calculate_vif_matrix(X):
    vif_matrix = pd.DataFrame(index=X.columns, columns=X.columns)
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i == j:
                vif_matrix.iloc[i, j] = 1.0
            else:
                X_temp = X.iloc[:, [i, j]]
                vif_matrix.iloc[i, j] = variance_inflation_factor(X_temp.values, 1)
    return vif_matrix

# Calculate the VIF matrix
vif_matrix = calculate_vif_matrix(X_train)
print(vif_matrix)