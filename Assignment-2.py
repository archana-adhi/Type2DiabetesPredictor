import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Step 1: Load and examine the data
# Load the dataset
df = pd.read_csv('diabetes.csv')

# EDA was done and features were analyzed in the jupyter notebook. Not repeating it here to keep the code and outputs direct & simple.

# Step 2: Data Preprocessing: Handle missing values, normalize/standardize the features, and split the data into training and test sets.
# Replace zeros with NaN in columns where zero is not biologically feasible
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Fill NaN values with the mean of each column
df.fillna(df.mean(), inplace=True)

# Separate features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 3: Model Implementation and Hyperparameter Tuning 
# SVM: Implement an SVM model, trying out different kernels and regularization parameters.
# Define the SVM model and grid search parameters
svm_model = SVC(random_state=42)
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']  # Parameter for kernel coefficient
}

# Perform grid search for the SVM model
svm_grid = GridSearchCV(estimator=svm_model, param_grid=svm_params, cv=3, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)

# XGBoost: Implement an XGBoost model, adjusting parameters like learning rate, max depth, and n_estimators.
# Define the XGBoost model and grid search parameters
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_params = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 6, 9],
    'n_estimators': [100, 200, 300]
}

# Perform grid search for the XGBoost model
xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train_scaled, y_train)

# Output the best parameters and scores for each model
print("SVM Best Parameters:", svm_grid.best_params_)
print("SVM Best Score:", svm_grid.best_score_,"\n")
print("XGBoost Best Parameters:", xgb_grid.best_params_)
print("XGBoost Best Score:", xgb_grid.best_score_,"\n")


# Step 4: Evaluation: Compare the models based on accuracy, precision, recall, and F1 score.
# Make predictions on the test set using the best models found by GridSearchCV
predictions_svm = svm_grid.predict(X_test_scaled)
predictions_xgb = xgb_grid.predict(X_test_scaled)

print("Classification report for SVM model :")
print(classification_report(y_test, predictions_svm))

print("Classification report for XGBoost model :")
print(classification_report(y_test, predictions_xgb ))

# Function to print evaluation metrics
def print_evaluation_metrics(y_true, y_pred, model_name):
    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_pred):.4f}\n")

# Print evaluation metrics for both models
print_evaluation_metrics(y_test, predictions_svm, 'SVM')
print_evaluation_metrics(y_test, predictions_xgb, 'XGBoost')