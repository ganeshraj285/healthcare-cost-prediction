import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
data = pd.read_csv(
    r"C:\Users\Admin\Desktop\Innomatics\Machine_Learning\Project_ML\Healthcare_insurance\healthcare_insurance_model_app\Data\insurance.csv"
)

# Keep only required columns
data = data[['age', 'bmi', 'smoker', 'charges']]

# Encode smoker
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

# Features & Target
X = data[['age', 'bmi', 'smoker']]
y = data['charges']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# ------------------ MODELS + HYPERPARAMETERS ------------------
models_params = {

    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    },

    "Decision Tree": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [2, 4, 6, 8, 10, None],
            "min_samples_split": [2, 5, 10]
        }
    },

    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4]
        }
    },

    "SVR": {
        "model": SVR(),
        "params": {
            "kernel": ['rbf', 'linear'],
            "C": [1, 10, 50],
            "gamma": ['scale', 'auto']
        }
    }
}

results = []
best_model = None
best_r2 = -999
best_model_name = ""

print("\n================= Hyperparameter Tuning + CV =================\n")

for name, mp in models_params.items():

    grid = GridSearchCV(
        estimator=mp["model"],
        param_grid=mp["params"],
        cv=kfold,
        scoring='r2',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    preds = best_estimator.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results.append([name, grid.best_params_, mae, mse, rmse, r2])

    print(f"📌 Model: {name}")
    print(f"Best Params: {grid.best_params_}")
    print(f"MAE  : {mae}")
    print(f"MSE  : {mse}")
    print(f"RMSE : {rmse}")
    print(f"R2   : {r2}")
    print("---------------------------------------------------")

    if r2 > best_r2:
        best_r2 = r2
        best_model = best_estimator
        best_model_name = name

# Results Summary
results_df = pd.DataFrame(results, columns=["Model", "Best Params", "MAE", "MSE", "RMSE", "R2 Score"])
print("\n============= FINAL RESULTS TABLE =============")
print(results_df)

print(f"\n🎯 Best Model Selected: {best_model_name}")
print(f"🏆 Best R2 Score: {best_r2}")

# Save Best Model
with open("insurance_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n✔️ Best Tuned Model Saved as insurance_model.pkl")
