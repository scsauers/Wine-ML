import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

file_path = 'https://raw.githubusercontent.com/scsauers/Wine-ML/main/winequality-red.csv'
print("Loading dataset...")
wine_data = pd.read_csv(file_path, delimiter=',')
print("Dataset loaded.")
print("Splitting data into features and target...")
X = wine_data.drop('quality', axis=1)
y = wine_data['quality'] - wine_data['quality'].min()  # Remap labels to start from 0
print("Data split completed.")

# Split the data into training and testing sets before scaling and PCA
print("Splitting data into training and testing sets...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print("Data split into training and testing sets.")





# Standardize the features - fit on training data only
print("Standardizing the features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
print("Features standardized.")

# Perform PCA to obtain principal components - fit on training data only
print("Performing PCA...")
pca = PCA(n_components=11)  # Number of components equal to the number of features
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("PCA completed.")

important_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Use only the selected principal components (PCs) for both training and testing sets
X_train_selected = X_train_pca[:, important_features]
X_test_selected = X_test_pca[:, important_features]
print("Selected PCs:", np.where(important_features)[0])






# Define and tune classifiers
# Random Forest
print("Defining Random Forest classifier and parameters...")
rf_classifier = RandomForestClassifier(random_state=123)
rf_param_grid = {
    'n_estimators': [10000],
    'max_depth': [None],
    'min_samples_split': [5],
    'min_samples_leaf': [1],
    'max_leaf_nodes': [None],
    'n_jobs': [-1],
    'warm_start': [True],
    'criterion': ['gini'],
}



# XGBoost
print("Defining XGBoost classifier and parameters...")
xgb_classifier = XGBClassifier(random_state=123, eval_metric='merror')
xgb_param_grid = {
    'n_estimators': [500],
    'max_depth': [None],
    'learning_rate': [0.1],
    'n_jobs': [-1]
}
print("XGBoost parameters defined.")

# SVC
print("Defining SVC classifier and parameters...")
svc_classifier = SVC(random_state=123)
svc_param_grid = {
    'C': [8],
    'gamma': [0.1],
    'kernel': ['rbf'],
}
print("SVC parameters defined.")


# KNN
print("Defining KNN classifier and parameters...")
knn_classifier = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': [11],
    'weights': ['distance'],
    'algorithm': ['auto'],
    'p': [1],
    'metric': ['minkowski'],
    'n_jobs': [-1]
}
print("KNN parameters defined.")



# XGBoost Grid Search
print("Performing XGBoost Grid Search...")
xgb_grid_search = GridSearchCV(xgb_classifier, xgb_param_grid, cv=5, scoring='accuracy')
xgb_grid_search.fit(X_train_selected, y_train)
print("Best Parameters (XGBoost):", xgb_grid_search.best_params_)
xgb_accuracy = accuracy_score(y_test, xgb_grid_search.best_estimator_.predict(X_test_selected))
print("Accuracy (XGBoost):", xgb_accuracy)

# SVC Grid Search
print("Performing SVC Grid Search...")
svc_grid_search = GridSearchCV(svc_classifier, svc_param_grid, cv=5, scoring='accuracy')
svc_grid_search.fit(X_train_selected, y_train)
print("Best Parameters (SVC):", svc_grid_search.best_params_)
svc_accuracy = accuracy_score(y_test, svc_grid_search.best_estimator_.predict(X_test_selected))
print("Accuracy (SVC):", svc_accuracy)


# Random Forest Grid Search
print("Performing Random Forest Grid Search...")
rf_grid_search = GridSearchCV(rf_classifier, rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train_selected, y_train)
print("Best Parameters (Random Forest):", rf_grid_search.best_params_)
rf_accuracy = accuracy_score(y_test, rf_grid_search.best_estimator_.predict(X_test_selected))
print("Accuracy (Random Forest):", rf_accuracy)

# KNN Grid Search
print("Performing KNN Grid Search...")
knn_grid_search = GridSearchCV(knn_classifier, knn_param_grid, cv=5, scoring='accuracy')
knn_grid_search.fit(X_train_selected, y_train)
print("Best Parameters (KNN):", knn_grid_search.best_params_)
knn_accuracy = accuracy_score(y_test, knn_grid_search.best_estimator_.predict(X_test_selected))
print("Accuracy (KNN):", knn_accuracy)
