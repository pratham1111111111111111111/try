import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------

df = pd.read_csv("traffic.csv")

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Create time-based features
df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek


# -------------------------------------------------
# 2. Features and Target
# -------------------------------------------------

features = [
    "Junction",
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek"
]

X = df[features]
y = df["Vehicles"]


# -------------------------------------------------
# 3. Train/Test Split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------------------------
# 4. MLflow Configuration
# -------------------------------------------------

mlflow.set_tracking_uri(
    "sqlite:///C:/Users/91775/Desktop/mlopss/mlflow.db"
)

mlflow.set_experiment("Traffic_Anomaly_Detection")


# -------------------------------------------------
# 5. Models
# -------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        random_state=42
    )
}


results = []


# -------------------------------------------------
# 6. Train Multiple Models
# -------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Prediction
        predictions = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        # Log parameters
        mlflow.log_param(
            "model",
            model_name
        )

        if model_name == "Random Forest":
            mlflow.log_param(
                "n_estimators",
                50
            )

        mlflow.log_param(
            "test_size",
            0.2
        )

        mlflow.log_param(
            "random_state",
            42
        )

        # Log metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        # Log model
        if model_name in ["Random Forest", "Decision Tree"]:
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )
        else:
            mlflow.sklearn.log_model(
                model,
                name="model"
            )

        # Get run ID
        run_id = mlflow.active_run().info.run_id

        # Save result
        results.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Run_ID": run_id
        })

        print("RMSE:", rmse)
        print("MAE :", mae)
        print("R2  :", r2)
        print("Run ID:", run_id)


# -------------------------------------------------
# 7. Performance Comparison
# -------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# Save comparison
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# -------------------------------------------------
# 8. Best Model Selection
# -------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

print("\nBest Model:")
print(best_model_name)


# Recreate best model
best_model = models[best_model_name]

best_model.fit(X_train, y_train)


# Save best model for FastAPI
joblib.dump(
    best_model,
    "models/traffic_model.joblib"
)

print("\nBest model saved to:")
print("models/traffic_model.joblib")


# -------------------------------------------------
# 9. Register Best Model in MLflow
# -------------------------------------------------

with mlflow.start_run(
    run_name="Best_Model_Selection"
):

    mlflow.log_param(
        "selected_model",
        best_model_name
    )

    mlflow.log_metric(
        "best_RMSE",
        results_df.iloc[0]["RMSE"]
    )

    mlflow.log_metric(
        "best_MAE",
        results_df.iloc[0]["MAE"]
    )

    mlflow.log_metric(
        "best_R2",
        results_df.iloc[0]["R2"]
    )

    if best_model_name in [
        "Random Forest",
        "Decision Tree"
    ]:
        mlflow.sklearn.log_model(
            best_model,
            name="best_model",
            registered_model_name="Traffic_Anomaly_Model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )
    else:
        mlflow.sklearn.log_model(
            best_model,
            name="best_model",
            registered_model_name="Traffic_Anomaly_Model"
        )

print("\nBest model registered in MLflow.")

print("\nExperiment completed successfully.")