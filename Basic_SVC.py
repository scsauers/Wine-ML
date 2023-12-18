import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.svm import SVC

file_path = '/Users/scott/Downloads/wine+quality/winequality-red.csv'
wine_data = pd.read_csv(file_path, delimiter=';')
wine_data['quality'] = wine_data['quality'] - 3
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
svc_classifier = SVC(random_state=42)
svc_cv_scores = cross_val_score(svc_classifier, X_train, y_train, cv=5, scoring=make_scorer(accuracy_score))
svc_cv_mean_accuracy = svc_cv_scores.mean()
print("Mean Accuracy (SVC):", svc_cv_mean_accuracy)
