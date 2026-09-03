import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, Input
from tensorflow.keras.optimizers import SGD
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

st.set_page_config(
    page_title="Network Traffic Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.2rem;}
.metric-card {
    padding: 16px; border-radius: 12px; background: white;
    border: 1px solid #e5e7eb; box-shadow: 0 2px 8px rgba(0,0,0,.04);
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Network Traffic Anomaly Detection")
st.caption("GRU-based sequential behavior modeling • prediction residuals used as an anomaly signal")

# ----------------------------------------------------------------------------
# Sidebar configuration
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded = st.file_uploader("Upload traffic CSV", type=["csv"])
    default_path = Path("traffic.csv")
    sequence_length = st.number_input("Sequence length", min_value=4, max_value=168, value=32, step=1)
    epochs = st.number_input("Training epochs", min_value=1, max_value=100, value=20, step=1)
    batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=150, step=8)
    threshold_percentile = st.slider("Anomaly threshold percentile", 90.0, 99.9, 95.0, 0.5)
    st.divider()
    run_comparison = st.checkbox("Compare GRU against baseline models", value=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(file_bytes=None):
    if file_bytes is not None:
        from io import BytesIO
        df = pd.read_csv(BytesIO(file_bytes))
    elif default_path.exists():
        df = pd.read_csv(default_path)
    else:
        return None
    required = {"DateTime", "Junction", "Vehicles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df["Junction"] = pd.to_numeric(df["Junction"], errors="coerce")
    df["Vehicles"] = pd.to_numeric(df["Vehicles"], errors="coerce")
    df = df.dropna(subset=["DateTime", "Junction", "Vehicles"]).copy()
    df = df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    return df

try:
    df = load_data(uploaded.getvalue() if uploaded else None)
except Exception as e:
    st.error(str(e))
    st.stop()

if df is None:
    st.info("Upload your traffic CSV from the sidebar, or place `traffic.csv` beside `app.py`.")
    st.stop()

junctions = sorted(df["Junction"].astype(int).unique().tolist())
with st.sidebar:
    selected_junction = st.selectbox("Junction", junctions)
    train_button = st.button("🚀 Train / Refresh GRU", use_container_width=True)

jdf = df[df["Junction"].astype(int) == int(selected_junction)].copy()
jdf = jdf.sort_values("DateTime").reset_index(drop=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(jdf):,}")
c2.metric("Junction", str(int(selected_junction)))
c3.metric("Avg vehicles", f"{jdf['Vehicles'].mean():.1f}")
c4.metric("Peak vehicles", f"{jdf['Vehicles'].max():.0f}")


# ----------------------------------------------------------------------------
# Sequence / feature helpers
# ----------------------------------------------------------------------------
def make_sequences(values, seq_len):
    x, y = [], []
    for i in range(seq_len, len(values)):
        x.append(values[i - seq_len:i])
        y.append(values[i])
    return np.asarray(x, dtype=np.float32).reshape(-1, seq_len, 1), np.asarray(y, dtype=np.float32)


def add_time_features(frame):
    out = frame.copy()
    out["hour"] = out["DateTime"].dt.hour
    out["dayofweek"] = out["DateTime"].dt.dayofweek
    out["month"] = out["DateTime"].dt.month
    out["dayofyear"] = out["DateTime"].dt.dayofyear
    return out


def build_gru(seq_len):
    model = Sequential([
        Input(shape=(seq_len, 1)),
        GRU(150, return_sequences=True, activation="tanh"),
        Dropout(0.2),
        GRU(150, return_sequences=True, activation="tanh"),
        Dropout(0.2),
        GRU(50, return_sequences=True, activation="tanh"),
        Dropout(0.2),
        GRU(50, return_sequences=True, activation="tanh"),
        Dropout(0.2),
        GRU(50, activation="tanh"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        loss="mean_squared_error",
    )
    return model


def train_and_score(values, dates, seq_len, n_epochs, bs):
    # Same basic sequence strategy as the supplied notebook:
    # first 90% training, last 10% testing, with a sequence length of 32 by default.
    split = int(len(values) * 0.90)
    train_values = values[:split]
    test_values = values[split:]

    # Normalize using training statistics; this keeps the test period unseen.
    mean = train_values.mean()
    std = train_values.std()
    if std == 0:
        std = 1.0

    train_scaled = (train_values - mean) / std
    test_scaled = (test_values - mean) / std

    X_train, y_train = make_sequences(train_scaled, seq_len)
    X_test, y_test = make_sequences(test_scaled, seq_len)

    if len(X_train) < 10 or len(X_test) < 1:
        raise ValueError("Not enough records for the selected sequence length.")

    model = build_gru(seq_len)
    early = tf.keras.callbacks.EarlyStopping(
        monitor="loss", min_delta=0.001, patience=10, restore_best_weights=True
    )

    with st.spinner("Training GRU model..."):
        model.fit(
            X_train, y_train,
            epochs=int(n_epochs),
            batch_size=int(bs),
            callbacks=[early],
            verbose=0,
        )

    pred_scaled = model.predict(X_test, verbose=0).reshape(-1)
    pred = pred_scaled * std + mean
    actual = y_test * std + mean
    test_dates = dates[split + seq_len:]

    result = pd.DataFrame({
        "DateTime": test_dates,
        "Actual": actual,
        "Predicted": pred,
    })
    result["Residual"] = result["Actual"] - result["Predicted"]
    result["Absolute_Error"] = result["Residual"].abs()

    threshold = np.percentile(result["Absolute_Error"], float(threshold_percentile))
    result["Anomaly"] = result["Absolute_Error"] > threshold
    return model, result, threshold, split


def score(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": r2_score(actual, predicted),
    }


def run_baseline_models(jframe, split, seq_len):
    """Fits fast baseline models on the same train/test split used for the
    GRU model, aligned to the same test window (split + seq_len onward),
    so all models are compared on identical ground truth."""
    feat_df = add_time_features(jframe)
    aligned = feat_df.iloc[split + seq_len:].copy()
    train_feat = feat_df.iloc[:split].copy()
    feature_cols = ["hour", "dayofweek", "month", "dayofyear"]

    actual = aligned["Vehicles"].to_numpy(dtype=np.float32)
    rows = {}

    # Naive / persistence baseline: predict the previous hour's value
    naive_pred = feat_df["Vehicles"].shift(1).iloc[split + seq_len:].to_numpy(dtype=np.float32)
    rows["Naive (persistence)"] = score(actual, naive_pred)

    # Moving average baseline (3-hour window)
    ma_pred = feat_df["Vehicles"].rolling(window=3, min_periods=1).mean().shift(1)
    ma_pred = ma_pred.iloc[split + seq_len:].to_numpy(dtype=np.float32)
    rows["Moving average (3h)"] = score(actual, ma_pred)

    # Linear Regression with calendar features
    lr = LinearRegression()
    lr.fit(train_feat[feature_cols], train_feat["Vehicles"])
    lr_pred = lr.predict(aligned[feature_cols])
    rows["Linear Regression"] = score(actual, lr_pred)

    # Random Forest with calendar features (small, fast config for the app)
    rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(train_feat[feature_cols], train_feat["Vehicles"])
    rf_pred = rf.predict(aligned[feature_cols])
    rows["Random Forest"] = score(actual, rf_pred)

    return rows


# ----------------------------------------------------------------------------
# Train only when the user clicks the button
# ----------------------------------------------------------------------------
settings_key = f"{selected_junction}_{sequence_length}_{epochs}_{batch_size}_{threshold_percentile}_{run_comparison}"

if "trained_key" not in st.session_state:
    st.session_state.trained_key = None

if train_button:
    try:
        values = jdf["Vehicles"].to_numpy(dtype=np.float32)
        dates = jdf["DateTime"].to_numpy()

        model, result, threshold, split = train_and_score(
            values, dates, sequence_length, epochs, batch_size
        )

        st.session_state.model = model
        st.session_state.result = result
        st.session_state.threshold = threshold
        st.session_state.split = split

        if run_comparison:
            with st.spinner("Training baseline models for comparison..."):
                baseline_scores = run_baseline_models(
                    jdf, split, sequence_length
                )
        else:
            baseline_scores = {}

        st.session_state.baseline_scores = baseline_scores
        st.session_state.trained_key = settings_key
        st.success("✅ GRU training completed.")
        st.rerun()

    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

# Only use a model result if it belongs to the current sidebar settings.
trained = (
    st.session_state.get("trained_key") == settings_key
    and "result" in st.session_state
)

if trained:
    model = st.session_state.model
    result = st.session_state.result
    threshold = st.session_state.threshold
    split = st.session_state.split
    baseline_scores = st.session_state.get("baseline_scores", {})

    anomalies = result[result["Anomaly"]].copy()
    rmse = float(np.sqrt(mean_squared_error(result["Actual"], result["Predicted"])))
    mae = float(mean_absolute_error(result["Actual"], result["Predicted"]))
    r2 = float(r2_score(result["Actual"], result["Predicted"]))
else:
    model = None
    result = pd.DataFrame()
    anomalies = pd.DataFrame()
    threshold = None
    split = None
    baseline_scores = {}
    rmse = None
    mae = None
    r2 = None

# ----------------------------------------------------------------------------
# Detection overview
# ----------------------------------------------------------------------------
if not trained:
    st.info(
        "👋 Your dataset and interactive charts are ready. "
        "The GRU model has not been trained yet. "
        "Click **🚀 Train / Refresh GRU** in the sidebar when you are ready."
    )
    o1, o2, o3 = st.columns(3)
    o1.metric("Dataset rows", f"{len(df):,}")
    o2.metric("Junctions", f"{len(junctions)}")
    o3.metric("Selected junction records", f"{len(jdf):,}")
else:
    st.subheader("📊 Detection Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Anomalies detected", f"{len(anomalies):,}")
    m2.metric("Anomaly rate", f"{100*len(anomalies)/max(len(result),1):.2f}%")
    m3.metric("RMSE", f"{rmse:.2f}")
    m4.metric("MAE", f"{mae:.2f}")

tab_data, tab_chart, tab_anom, tab_compare, tab_event, tab_metrics = st.tabs([
    "📂 Dataset View",
    "📈 Traffic & Predictions",
    "🚨 Anomalies",
    "🏆 Model Comparison",
    "🔎 Single Event",
    "📋 Data & Metrics",
])

# ----------------------------------------------------------------------------
# Dataset View
# ----------------------------------------------------------------------------
with tab_data:
    st.write("### Dataset overview")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total rows (all junctions)", f"{len(df):,}")
    d2.metric("Junctions", f"{len(junctions)}")
    d3.metric("Date range start", df["DateTime"].min().strftime("%Y-%m-%d"))
    d4.metric("Date range end", df["DateTime"].max().strftime("%Y-%m-%d"))

    left, right = st.columns([1, 1])
    with left:
        st.write("#### Column summary")
        summary = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "unique": df.nunique(),
        })
        st.dataframe(summary, use_container_width=True)
    with right:
        st.write("#### Descriptive statistics (Vehicles)")
        st.dataframe(df.groupby("Junction")["Vehicles"].describe(), use_container_width=True)

    st.write("#### Records per junction")
    counts = df["Junction"].astype(int).value_counts().sort_index().reset_index()
    counts.columns = ["Junction", "Records"]
    fig_counts = px.bar(counts, x="Junction", y="Records", color="Junction",
                         title="Record count by junction", text="Records")
    fig_counts.update_layout(showlegend=False)
    st.plotly_chart(fig_counts, use_container_width=True)

    st.write("#### Average traffic by hour of day × day of week (selected junction)")
    heat = add_time_features(jdf)
    pivot = heat.pivot_table(index="dayofweek", columns="hour", values="Vehicles", aggfunc="mean")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig_heat = px.imshow(
        pivot.values, x=pivot.columns, y=[day_labels[i] for i in pivot.index],
        aspect="auto", color_continuous_scale="YlOrRd",
        labels=dict(x="Hour of day", y="Day of week", color="Avg vehicles"),
        title=f"Junction {int(selected_junction)} — average traffic heatmap",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.write("#### Raw data preview")
    n_rows = st.slider("Rows to preview", 10, 500, 100, 10)
    st.dataframe(jdf.head(n_rows), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download this junction's data",
        jdf.to_csv(index=False).encode("utf-8"),
        file_name=f"junction_{int(selected_junction)}_data.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
# Traffic & Predictions (interactive)
# ----------------------------------------------------------------------------
with tab_chart:
    if not trained:
        st.info("Train the GRU from the sidebar to view predictions and anomaly signals.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result["DateTime"], y=result["Actual"],
            mode="lines", name="Actual", line=dict(width=1.4)
        ))
        fig.add_trace(go.Scatter(
            x=result["DateTime"], y=result["Predicted"],
            mode="lines", name="GRU Prediction",
            line=dict(width=1.2, dash="dot")
        ))
        if len(anomalies):
            fig.add_trace(go.Scatter(
                x=anomalies["DateTime"], y=anomalies["Actual"],
                mode="markers", name="Anomaly",
                marker=dict(color="red", size=8, symbol="x"),
                hovertemplate="Anomaly<br>%{x}<br>Vehicles: %{y}<extra></extra>",
            ))

        fig.update_layout(
            title=f"Junction {int(selected_junction)} — Sequential Behavior vs GRU Prediction",
            xaxis_title="DateTime",
            yaxis_title="Vehicles / Traffic Volume",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                ),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"An event is flagged when the absolute GRU prediction error exceeds "
            f"the {threshold_percentile:.1f}th-percentile threshold "
            f"({threshold:.2f} vehicles)."
        )

        st.write("#### Residual distribution")
        fig_resid = px.histogram(
            result, x="Absolute_Error", nbins=40,
            title="Distribution of absolute prediction error"
        )
        fig_resid.add_vline(
            x=threshold, line_dash="dash", line_color="red",
            annotation_text="Anomaly threshold"
        )
        st.plotly_chart(fig_resid, use_container_width=True)

# ----------------------------------------------------------------------------
# Anomalies
# ----------------------------------------------------------------------------
with tab_anom:
    if not trained:
        st.info("Train the GRU from the sidebar to detect anomalies.")
    elif len(anomalies):
        display_cols = ["DateTime", "Actual", "Predicted", "Residual", "Absolute_Error"]
        st.dataframe(
            anomalies[display_cols].sort_values("Absolute_Error", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        csv = anomalies[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download anomaly report",
            csv,
            file_name=f"junction_{int(selected_junction)}_anomalies.csv",
            mime="text/csv",
        )

        fig_scatter = px.scatter(
            result, x="DateTime", y="Absolute_Error", color="Anomaly",
            color_discrete_map={True: "red", False: "#93c5fd"},
            title="Absolute error over time (flagged points in red)",
        )
        fig_scatter.add_hline(
            y=threshold, line_dash="dash", line_color="black",
            annotation_text="Threshold"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.success("No anomalies were detected at the current threshold.")

# ----------------------------------------------------------------------------
# Model Comparison
# ----------------------------------------------------------------------------
with tab_compare:
    if not trained:
        st.info("Train the GRU from the sidebar to compare model performance.")
    elif not run_comparison:
        st.info(
            "Enable **Compare GRU against baseline models** in the sidebar, "
            "then retrain to populate this tab."
        )
    else:
        st.write("### GRU vs. baseline models")
        st.caption(
            "All models are evaluated on the identical held-out test window "
            "(last 10% of the series, after the GRU's initial sequence warm-up)."
        )

        rows = {
            "GRU (sequential behavior model)": {
                "MAE": mae, "RMSE": rmse, "R2": r2
            }
        }
        rows.update(baseline_scores)
        comp_df = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "Model"})
        comp_df = comp_df.sort_values("RMSE").reset_index(drop=True)

        st.dataframe(
            comp_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"})
                     .highlight_min(subset=["MAE", "RMSE"], color="#d1fae5")
                     .highlight_max(subset=["R2"], color="#d1fae5"),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_rmse = px.bar(
                comp_df, x="Model", y="RMSE", color="Model",
                title="RMSE by model (lower is better)", text_auto=".2f"
            )
            fig_rmse.update_layout(showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)

        with c2:
            fig_r2 = px.bar(
                comp_df, x="Model", y="R2", color="Model",
                title="R² by model (higher is better)", text_auto=".3f"
            )
            fig_r2.update_layout(showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)

        best_model = comp_df.iloc[0]["Model"]
        st.success(f"🏆 Best model on this test window (by RMSE): **{best_model}**")

# ----------------------------------------------------------------------------
# Single Event
# ----------------------------------------------------------------------------
with tab_event:
    if not trained:
        st.info("Train the GRU from the sidebar to inspect individual test events.")
    elif len(result):
        idx = st.number_input(
            "Select test event index",
            min_value=0, max_value=len(result) - 1, value=0, step=1
        )
        row = result.iloc[int(idx)]
        a, b, c = st.columns(3)
        a.metric("Actual", f"{row['Actual']:.2f}")
        b.metric("GRU prediction", f"{row['Predicted']:.2f}")
        c.metric("Absolute error", f"{row['Absolute_Error']:.2f}")

        if bool(row["Anomaly"]):
            st.error(
                "🚨 ANOMALY: sequential traffic behavior differs strongly "
                "from the GRU expectation."
            )
        else:
            st.success(
                "✅ NORMAL: behavior is within the learned residual range."
            )

# ----------------------------------------------------------------------------
# Data & Metrics
# ----------------------------------------------------------------------------
with tab_metrics:
    if not trained:
        st.info("Train the GRU from the sidebar to view model metrics and test predictions.")
    else:
        st.write("### Model performance on the final 10% test period")
        metrics_df = pd.DataFrame({
            "Metric": ["MAE", "MSE", "RMSE", "R²", "Threshold"],
            "Value": [
                mae,
                float(mean_squared_error(result["Actual"], result["Predicted"])),
                rmse,
                r2,
                float(threshold),
            ],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.write("### Test predictions")
        st.dataframe(result, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Architecture follows the GRU structure in the supplied notebook: "
    "150 → 150 → 50 → 50 → 50 GRU units with 0.2 dropout and a Dense(1) output. "
    "Baseline comparisons: naive persistence, 3-hour moving average, linear regression, "
    "and random forest, each trained on calendar features over the same split."
)