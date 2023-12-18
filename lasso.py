import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score

file_path = '/Users/scott/Downloads/wine+quality/winequality-red.csv'
wine_data = pd.read_csv(file_path, delimiter=';')
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2023)
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
y_pred_class = np.rint(y_pred)
accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy (Lasso Regression as Classifier):", accuracy)
