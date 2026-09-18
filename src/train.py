import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("traffic.csv")

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ==========================================
# 2. DATETIME CONVERSION
# ==========================================

df["DateTime"] = pd.to_datetime(df["DateTime"])


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

X = df[
    [
        "Junction",
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek"
    ]
]

y = df["Vehicles"]


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 6. MLflow EXPERIMENT
# ==========================================

mlflow.set_experiment("Traffic_Anomaly_Detection")


# ==========================================
# 7. START MLflow RUN
# ==========================================

with mlflow.start_run(run_name="Random_Forest"):

    # ======================================
    # 8. MODEL PARAMETERS
    # ======================================

    n_estimators = 50
    random_state = 42


    # ======================================
    # 9. CREATE MODEL
    # ======================================

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )


    # ======================================
    # 10. TRAIN MODEL
    # ======================================

    model.fit(X_train, y_train)


    # ======================================
    # 11. PREDICTION
    # ======================================

    predictions = model.predict(X_test)


    # ======================================
    # 12. METRICS
    # ======================================

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


    # ======================================
    # 13. PRINT RESULTS
    # ======================================

    print("\nModel Training Completed")
    print("------------------------")
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2 Score:", r2)


    # ======================================
    # 14. LOG PARAMETERS
    # ======================================

    mlflow.log_param(
        "model_type",
        "RandomForestRegressor"
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "test_size",
        0.2
    )


    # ======================================
    # 15. LOG METRICS
    # ======================================

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)


    # ======================================
    # 16. SAVE LOCAL MODEL
    # ======================================

    os.makedirs("models", exist_ok=True)

    model_path = "models/traffic_model.joblib"

    joblib.dump(
        model,
        model_path
    )

    print("\nLocal model saved:")
    print(model_path)


    # ======================================
    # 17. LOG + REGISTER MODEL
    # ======================================

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="traffic_model",
        registered_model_name="Traffic_Anomaly_Model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )


    # ======================================
    # 18. ALSO SAVE LOCAL MODEL AS ARTIFACT
    # ======================================

    mlflow.log_artifact(
        model_path,
        artifact_path="local_model"
    )


    # ======================================
    # 19. FINAL INFORMATION
    # ======================================

    run_id = mlflow.active_run().info.run_id

    print("\n================================")
    print("MLflow Tracking Completed")
    print("================================")
    print("Experiment: Traffic_Anomaly_Detection")
    print("Run Name: Random_Forest")
    print("Run ID:", run_id)
    print("Model Name: Traffic_Anomaly_Model")
    print("Model registration requested.")