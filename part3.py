import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pickle

# Load dataset
file_path = "../datasets/part3_dataset.data"
with open(file_path, 'rb') as file:
    dataset, labels = pickle.load(file)

X, y = dataset, labels

# Define algorithms and their hyperparameter grids
models = {
    "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5]}),
    "SVM": (SVC(), {"C": [0.1, 1], "kernel": ["linear", "rbf"]}),
    "DecisionTree": (DecisionTreeClassifier(), {"max_depth": [5, 10]}),
    "RandomForest": (RandomForestClassifier(), {"n_estimators": [50, 100]}),
    "MLP": (MLPClassifier(max_iter=1000, early_stopping=True, random_state=42), {"hidden_layer_sizes": [(50,), (100,)]}),
    "GradientBoosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100]})
}

# Nested cross-validation setup
outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
f1_scorer = make_scorer(f1_score, average="weighted")

results = {}

for name, (model, param_grid) in models.items():
    print(f"Evaluating {name}...")
    outer_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=f1_scorer, cv=inner_cv)
        grid_search.fit(X_train, y_train)

        # Evaluate on outer fold
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        outer_scores.append(f1_score(y_test, y_pred, average="weighted"))

    # Store results
    results[name] = {
        "mean_f1": np.mean(outer_scores),
        "std_f1": np.std(outer_scores),
        "confidence_interval": (np.mean(outer_scores) - 1.96 * np.std(outer_scores) / np.sqrt(len(outer_scores)),
                                  np.mean(outer_scores) + 1.96 * np.std(outer_scores) / np.sqrt(len(outer_scores)))
    }

# Display results
results_df = pd.DataFrame(results).T
print(results_df)

# Save results to a CSV file
results_df.to_csv("classification_results.csv", index=True)
