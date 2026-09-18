import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. MLflow Configuration
# ============================================================

# Local Windows MLflow database is used by default.
# GitHub Actions can override this using MLFLOW_TRACKING_URI.

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///C:/Users/91775/Desktop/mlopss/mlflow.db"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "Traffic_Anomaly_Detection"

mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 60)
print("TRAFFIC ANOMALY DETECTION - MULTI MODEL EXPERIMENT")
print("=" * 60)

print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"MLflow Experiment: {EXPERIMENT_NAME}")


# ============================================================
# 2. Load Dataset
# ============================================================

DATA_PATH = "traffic.csv"

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 3. Data Preprocessing
# ============================================================

print("\nPreprocessing data...")

# Convert DateTime column to datetime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Create time-based features
df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek


# Features
FEATURES = [
    "Junction",
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek"
]

TARGET = "Vehicles"

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 4. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 5. Define Multiple Models
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 6. Train and Evaluate Models
# ============================================================

results = []

print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION")
print("=" * 60)


for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    # Start MLflow run
    with mlflow.start_run(run_name=model_name):

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Metrics
        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES)
        )

        mlflow.log_param(
            "test_size",
            0.20
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # Log model-specific parameters
        if model_name == "Random Forest":
            mlflow.log_param(
                "n_estimators",
                50
            )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "rmse",
            rmse
        )

        mlflow.log_metric(
            "mae",
            mae
        )

        mlflow.log_metric(
            "r2",
            r2
        )

        # ----------------------------------------------------
        # Log model to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # Store results
        results.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        print(f"RMSE: {rmse}")
        print(f"MAE : {mae}")
        print(f"R2  : {r2}")


# ============================================================
# 7. Model Performance Comparison
# ============================================================

results_df = pd.DataFrame(results)

# Lower RMSE is better
results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ============================================================
# 8. Save Comparison Results
# ============================================================

os.makedirs("models", exist_ok=True)

comparison_path = "models/model_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False
)

print(f"\nModel comparison saved to: {comparison_path}")


# ============================================================
# 9. Select Best Model
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

print("\n" + "=" * 60)
print("BEST MODEL SELECTION")
print("=" * 60)

print(f"Best Model: {best_model_name}")


# Get best model object
best_model = models[best_model_name]


# ============================================================
# 10. Save Best Model
# ============================================================

best_model_path = "models/traffic_model.joblib"

joblib.dump(
    best_model,
    best_model_path
)

print(f"Best model saved to: {best_model_path}")


# ============================================================
# 11. Register Best Model in MLflow
# ============================================================

print("\nRegistering best model in MLflow...")

with mlflow.start_run(
    run_name="Best_Model_Registration"
):

    mlflow.log_param(
        "selected_model",
        best_model_name
    )

    mlflow.log_metric(
        "best_rmse",
        float(results_df.iloc[0]["RMSE"])
    )

    mlflow.log_metric(
        "best_mae",
        float(results_df.iloc[0]["MAE"])
    )

    mlflow.log_metric(
        "best_r2",
        float(results_df.iloc[0]["R2"])
    )

    mlflow.sklearn.log_model(
        best_model,
        name="best_model",
        registered_model_name="Traffic_Anomaly_Model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )


# ============================================================
# 12. Final Output
# ============================================================

print("\n" + "=" * 60)
print("AUTOMATIC MODEL TRAINING COMPLETED")
print("=" * 60)

print(f"Best Model : {best_model_name}")
print(
    f"Best RMSE  : {results_df.iloc[0]['RMSE']}"
)
print(
    f"Best MAE   : {results_df.iloc[0]['MAE']}"
)
print(
    f"Best R2    : {results_df.iloc[0]['R2']}"
)

print(f"Model File : {best_model_path}")
print(f"Comparison : {comparison_path}")

print("\nMLflow experiment completed successfully.")
print("=" * 60)