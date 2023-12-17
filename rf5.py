from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the wine dataset
file_path = r"C:\Users\sluca\Downloads\Wine Data.xlsx"
wine_data = pd.read_excel(file_path)

# Separate features (X) and target variable (y)
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define a reduced parameter grid for randomized search
param_dist_reduced = {
    'n_estimators': randint(50, 150),
    'max_depth': [None] + list(range(10, 60, 10)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Perform randomized search with 5-fold cross-validation in parallel
random_search_reduced = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist_reduced,
                                           n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search_reduced.fit(X_train, y_train)

# Get the best parameters from the randomized search
best_params_reduced = random_search_reduced.best_params_

# Use the best parameters to train the classifier
best_rf_classifier_reduced = RandomForestClassifier(random_state=42, **best_params_reduced)
best_rf_classifier_reduced.fit(X_train, y_train)

# Make predictions on the test data
y_pred_best_reduced = best_rf_classifier_reduced.predict(X_test)

# Evaluate the performance of the tuned classifier
accuracy_best_reduced = accuracy_score(y_test, y_pred_best_reduced)
conf_matrix_best_reduced = confusion_matrix(y_test, y_pred_best_reduced)
class_report_best_reduced = classification_report(y_test, y_pred_best_reduced)

# Print the results
print(f'Best Hyperparameters (Reduced Search): {best_params_reduced}\n')
print(f'Accuracy with Best Hyperparameters (Reduced Search): {accuracy_best_reduced:.4f}\n')
print(f'Confusion Matrix with Best Hyperparameters (Reduced Search):\n{conf_matrix_best_reduced}\n')
print(f'Classification Report with Best Hyperparameters (Reduced Search):\n{class_report_best_reduced}')
