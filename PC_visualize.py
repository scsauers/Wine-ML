import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/scott/Downloads/wine+quality/winequality-red.csv'
wine_data = pd.read_csv(file_path, delimiter=';')

X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca_2 = PCA(n_components=2)
principalComponents_2 = pca_2.fit_transform(X_scaled)
pca_df_2 = pd.DataFrame(data=principalComponents_2, columns=['PC1', 'PC2'])
pca_df_2['quality'] = y

pca_max = PCA(n_components=min(X_scaled.shape[1], 20))
principalComponents_max = pca_max.fit_transform(X_scaled)
explained_variance_max = pca_max.explained_variance_ratio_
correlation_with_quality_max = [np.corrcoef(principalComponents_max[:, i], y)[0, 1] for i in range(pca_max.n_components_)]

pc_max_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(pca_max.n_components_)],
    'Explained Variance': explained_variance_max,
    'Correlation with Quality': correlation_with_quality_max
})

plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df_2, x='PC1', y='PC2', hue='quality', palette='viridis')
plt.title('PCA of Wine Quality Dataset (2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='PC', y='Explained Variance', data=pc_max_df)
plt.title('Explained Variance by Principal Components')
plt.xticks(rotation=45)
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='PC', y='Correlation with Quality', data=pc_max_df)
plt.title('Correlation of Principal Components with Wine Quality')
plt.xticks(rotation=45)
plt.ylabel('Correlation Coefficient')
plt.xlabel('Principal Components')
plt.show()
