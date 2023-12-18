import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Provided confusion matrix
confusion_matrix_data = np.array([[0, 0, 1, 0, 0, 0],
                                  [0, 0, 7, 3, 0, 0],
                                  [0, 0, 100, 29, 1, 0],
                                  [0, 0, 32, 93, 7, 0],
                                  [0, 0, 0, 19, 23, 0],
                                  [0, 0, 0, 1, 4, 0]])

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.arange(1, 7), yticklabels=np.arange(1, 7))
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix Heatmap')
plt.show()
