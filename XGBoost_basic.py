import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer
from xgboost import XGBClassifier

file_path = 'https://raw.githubusercontent.com/scsauers/Wine-ML/main/winequality-red.csv'
print("Loading dataset...")
wine_data = pd.read_csv(file_path, delimiter=',')
print("Dataset loaded.")

# remapping quality scores, 3-8 to 0-5
wine_data['quality'] = wine_data['quality'] - 3

X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2023)
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=2023)
xgb_cv_scores = cross_val_score(xgb_classifier, X_train, y_train, cv=5, scoring=make_scorer(accuracy_score))
xgb_cv_mean_accuracy = xgb_cv_scores.mean()
print("Mean Accuracy (XGBoost):", xgb_cv_mean_accuracy)
