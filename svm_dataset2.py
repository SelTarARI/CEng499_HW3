import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
dataset, labels = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

# Define hyperparameter configurations to test
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Define performance metric
scoring_metric = 'accuracy'

# Prepare to collect results
results = []

# Repeat cross-validation 5 times with shuffling
for i in range(5):
    print(f"Iteration {i + 1}/5")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
    clf = GridSearchCV(
        estimator=SVC(),
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=skf,
        return_train_score=False,
        n_jobs=-1
    )

    # Preprocessing inside CV loop
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)  # Scale the entire dataset

    # Fit the model
    clf.fit(dataset_scaled, labels)

    # Store results
    for params, mean_score, std_score in zip(
        clf.cv_results_['params'],
        clf.cv_results_['mean_test_score'],
        clf.cv_results_['std_test_score']
    ):
        results.append({
            'iteration': i + 1,
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Convert 'params' column to a string for grouping
results_df['params_str'] = results_df['params'].apply(str)

# Group by the string representation of the parameters
grouped = results_df.groupby('params_str').agg(
    mean_score=('mean_score', 'mean'),
    std_score=('mean_score', 'std')
)

# Calculate confidence intervals
confidence_intervals = grouped.assign(
    lower_bound=grouped['mean_score'] - 1.96 * grouped['std_score'] / np.sqrt(5),
    upper_bound=grouped['mean_score'] + 1.96 * grouped['std_score'] / np.sqrt(5)
)

# Save results
print("Final results:")
print(confidence_intervals)
confidence_intervals.to_csv("svm_dataset2_results.csv")
