import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import random
from joblib import Parallel, delayed

import warnings
from sklearn.exceptions import ConvergenceWarning

# suppress ConvergenceWarning in sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)

file_path = 'https://raw.githubusercontent.com/scsauers/Wine-ML/main/winequality-red.csv'
wine_data = pd.read_csv(file_path, delimiter=',')
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Scale the test set

# Function to make an individual
def create_individual():
    model_choice = random.choice(['rf', 'lr', 'knn'])
    if model_choice == 'rf':
        n_estimators = random.choice([10, 50, 100])
        return {'model': 'rf', 'n_estimators': n_estimators}
    elif model_choice == 'lr':
        max_iter = random.choice([100, 200, 300])
        return {'model': 'lr', 'max_iter': max_iter}
    else:
        n_neighbors = random.choice([3, 5, 7])
        return {'model': 'knn', 'n_neighbors': n_neighbors}


# Mutate an individual
def mutate(individual):
    if individual['model'] == 'rf':
        individual['n_estimators'] = random.choice([10, 50, 100])
    elif individual['model'] == 'lr':
        individual['max_iter'] = random.choice([100, 200, 300])
    else:
        individual['n_neighbors'] = random.choice([3, 5, 7])
    return individual

# Evaluate using cross-validation
def evaluate(individual):
    if individual['model'] == 'rf':
        model = RandomForestClassifier(n_estimators=individual['n_estimators'])
    elif individual['model'] == 'lr':
        model = LogisticRegression(max_iter=individual['max_iter'])
    else:
        model = KNeighborsClassifier(n_neighbors=individual['n_neighbors'])
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    return np.mean(scores)


# Execute
population_size = 100
population = [create_individual() for _ in range(population_size)]
fitness_values = []

for gen in range(5):
    fitness_scores = Parallel(n_jobs=-1)(delayed(evaluate)(ind) for ind in population)
    max_fitness = max(fitness_scores)
    fitness_values.append(max_fitness)

    print(f"Generation {gen + 1}, Max Fitness: {max_fitness:.6f}")

    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    population = sorted_population[:len(sorted_population) // 2]

    offspring = [mutate(ind.copy()) for ind in population]
    population.extend(offspring)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(fitness_values) + 1), fitness_values, marker='o')
plt.xlabel('Generation')
plt.ylabel('Fitness (Accuracy)')
plt.title('Fitness over Generations')
plt.show()

# test
best_individual = sorted_population[0]
if best_individual['model'] == 'rf':
    best_model = RandomForestClassifier(n_estimators=best_individual['n_estimators'])
elif best_individual['model'] == 'lr':
    best_model = LogisticRegression(max_iter=best_individual['max_iter'])
else:
    best_model = KNeighborsClassifier(n_neighbors=best_individual['n_neighbors'])

best_model.fit(X_train_scaled, y_train)
test_accuracy = best_model.score(X_test_scaled, y_test)
print(f"Final Top Fitness: {max(fitness_values)}")
print(f"Test Set Accuracy: {test_accuracy}")
