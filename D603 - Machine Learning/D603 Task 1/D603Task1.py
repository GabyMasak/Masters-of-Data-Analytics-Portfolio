# Gaby Masak
# D603 - Machine Learning
# Task 1

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Define the file path
file_path = r"C:\Users\gabri\OneDrive\Documents\Education\WGU\MSDA\D603 - Machine Learning\D603 Task 1\medical_clean.csv"

# Read the CSV file into a DataFrame
medData = pd.read_csv(file_path)

# Drop duplicate rows
medData.drop_duplicates(inplace=True)

# Drop unnecessary columns
medData.drop(columns=['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State', 'County', 'Zip', 'TimeZone'], inplace=True)

# Handle missing values
for column in medData.columns:
    if medData[column].dtype == 'object':
        medData[column] = medData[column].fillna(medData[column].mode()[0])
    else:
        medData[column] = medData[column].fillna(medData[column].mean())

# Identify binary columns
binary_columns = [col for col in medData.columns if medData[col].nunique() == 2]

# Identify other categorical columns
categorical_columns = [col for col in medData.columns if medData[col].dtype == 'object' and col not in binary_columns]

# Create a column transformer with OneHotEncoder for binary and other categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('binary', OneHotEncoder(drop='first'), binary_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_columns)
    ],
    remainder='passthrough'
)

# Apply the column transformer to the entire dataset
medData_encoded = preprocessor.fit_transform(medData).toarray()
# Get the feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# Convert the encoded data back to a DataFrame
medData_encoded = pd.DataFrame(medData_encoded, columns=feature_names)

# Export the cleaned and encoded data to a CSV file
medData_encoded.to_csv('cleaned_data.csv', index=False)

print("Cleaned and encoded data has been exported to 'cleaned_data.csv'")

# # Print the column names to identify the target column
# print(medData_encoded.columns)

# Split the data into features (X) and target (y)
# Adjust the column name based on the printed column names
X = medData_encoded.drop(columns=['binary__ReAdmis_Yes'])
y = medData_encoded['binary__ReAdmis_Yes']

# Split the data into training, validation, and test datasets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize the Gradient Boosting model
model = GradientBoostingClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]  # Get the predicted probabilities for the positive class

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc_roc = roc_auc_score(y_val, y_pred_proba)
conf_matrix = confusion_matrix(y_val, y_pred)

# Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC: {auc_roc}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform the grid search
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Train the optimized model
optimized_model = grid_search.best_estimator_

# Make predictions on the validation set
y_pred_optimized = optimized_model.predict(X_val)
y_pred_proba_optimized = optimized_model.predict_proba(X_val)[:, 1]  # Get the predicted probabilities for the positive class

# Calculate metrics for the optimized model
accuracy_optimized = accuracy_score(y_val, y_pred_optimized)
precision_optimized = precision_score(y_val, y_pred_optimized)
recall_optimized = recall_score(y_val, y_pred_optimized)
f1_optimized = f1_score(y_val, y_pred_optimized)
auc_roc_optimized = roc_auc_score(y_val, y_pred_proba_optimized)
conf_matrix_optimized = confusion_matrix(y_val, y_pred_optimized)

# Print metrics for the optimized model
print(f"Optimized Accuracy: {accuracy_optimized}")
print(f"Optimized Precision: {precision_optimized}")
print(f"Optimized Recall: {recall_optimized}")
print(f"Optimized F1 Score: {f1_optimized}")
print(f"Optimized AUC-ROC: {auc_roc_optimized}")

# Plot confusion matrix for the optimized model
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_optimized, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Optimized Confusion Matrix')
plt.show()

# Make predictions on the test set
y_test_pred = optimized_model.predict(X_test)
y_test_pred_proba = optimized_model.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for the positive class

# Calculate metrics for the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
auc_roc_test = roc_auc_score(y_test, y_test_pred_proba)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Print metrics for the test set
print(f"Test Accuracy: {accuracy_test}")
print(f"Test Precision: {precision_test}")
print(f"Test Recall: {recall_test}")
print(f"Test F1 Score: {f1_test}")
print(f"Test AUC-ROC: {auc_roc_test}")

# Plot confusion matrix for the test set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Test Confusion Matrix')
plt.show()
