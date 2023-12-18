import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import itertools

file_path = 'https://raw.githubusercontent.com/scsauers/Wine-ML/main/winequality-red.csv'
print("Loading dataset...")
wine_data = pd.read_csv(file_path, delimiter=',')
print("Dataset loaded.")
print("Splitting data into features and target...")
X = wine_data.drop('quality', axis=1)
y = wine_data['quality'] - wine_data['quality'].min()
print("Data split completed.")
print("Splitting data into training and testing sets...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print("Data split into training and testing sets.")
print("Standardizing the features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
print("Features standardized.")
print("Performing PCA...")
pca = PCA(n_components=11)  # 1-based indexing... components equal to the number of features
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("PCA completed.")

important_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X_train_selected = X_train_pca[:, important_features]
X_test_selected = X_test_pca[:, important_features]
print("Selected PCs:", np.where(important_features)[0])
def create_interaction_terms(data):
    n_features = data.shape[1]
    interaction_terms = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            interaction_terms.append(data[:, i] * data[:, j])
    return np.array(interaction_terms).T
X_train_interactions = create_interaction_terms(X_train_selected)
X_test_interactions = create_interaction_terms(X_test_selected)
print("Number of interaction terms:", X_train_interactions.shape[1])
X_train_combined = np.hstack((X_train_selected, X_train_interactions))
X_test_combined = np.hstack((X_test_selected, X_test_interactions))

# KNN
print("Defining KNN classifier and parameters...")
knn_classifier = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': [10, 11, 12],          # Different vals for number of neighbors
    'weights': ['distance'],
    'algorithm': ['auto'],
    'p': [1, 2, 2.5, 3, 4],  # Manhattan (p=1) and Euclidean (p=2) distances... others
    'metric': ['minkowski', 'euclidean', 'manhattan'],
    'n_jobs': [-1]
}
print("KNN parameters defined.")

# KNN grid search
print("Performing KNN Grid Search...")
knn_grid_search = GridSearchCV(knn_classifier, knn_param_grid, cv=5, scoring='accuracy')
knn_grid_search.fit(X_train_combined, y_train)
print("Best Parameters (KNN):", knn_grid_search.best_params_)
knn_accuracy = accuracy_score(y_test, knn_grid_search.predict(X_test_combined))
print("Accuracy (KNN):", knn_accuracy)
