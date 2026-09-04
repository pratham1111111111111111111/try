
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, Input
from tensorflow.keras.optimizers import SGD
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

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded = st.file_uploader("Upload traffic CSV", type=["csv"])
    default_path = Path("traffic.csv")
    sequence_length = st.number_input("Sequence length", min_value=4, max_value=168, value=32, step=1)
    epochs = st.number_input("Training epochs", min_value=1, max_value=100, value=20, step=1)
    batch_size = st.number_input("Batch size", min_value=8, max_value=512, value=150, step=8)
    threshold_percentile = st.slider("Anomaly threshold percentile", 90, 99.9, 95.0, 0.5)

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

def make_sequences(values, seq_len):
    x, y = [], []
    for i in range(seq_len, len(values)):
        x.append(values[i-seq_len:i])
        y.append(values[i])
    return np.asarray(x, dtype=np.float32).reshape(-1, seq_len, 1), np.asarray(y, dtype=np.float32)

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
    return model, result, threshold

# Train automatically once per junction/setting; button clears cached session result.
settings_key = f"{selected_junction}_{sequence_length}_{epochs}_{batch_size}_{threshold_percentile}"
if "trained_key" not in st.session_state or train_button or st.session_state.get("trained_key") != settings_key:
    try:
        values = jdf["Vehicles"].to_numpy(dtype=np.float32)
        dates = jdf["DateTime"].to_numpy()
        model, result, threshold = train_and_score(
            values, dates, sequence_length, epochs, batch_size
        )
        st.session_state.model = model
        st.session_state.result = result
        st.session_state.threshold = threshold
        st.session_state.trained_key = settings_key
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()
else:
    model = st.session_state.model
    result = st.session_state.result
    threshold = st.session_state.threshold

anomalies = result[result["Anomaly"]].copy()
rmse = float(np.sqrt(mean_squared_error(result["Actual"], result["Predicted"])))
mae = float(mean_absolute_error(result["Actual"], result["Predicted"]))
r2 = float(r2_score(result["Actual"], result["Predicted"]))

st.subheader("📊 Detection Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Anomalies detected", f"{len(anomalies):,}")
m2.metric("Anomaly rate", f"{100*len(anomalies)/max(len(result),1):.2f}%")
m3.metric("RMSE", f"{rmse:.2f}")
m4.metric("MAE", f"{mae:.2f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Traffic & Predictions",
    "🚨 Anomalies",
    "🔎 Single Event",
    "📋 Data & Metrics"
])

with tab1:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(result["DateTime"], result["Actual"], label="Actual", linewidth=1.2)
    ax.plot(result["DateTime"], result["Predicted"], label="GRU Prediction", linewidth=1.0)
    if len(anomalies):
        ax.scatter(anomalies["DateTime"], anomalies["Actual"], label="Anomaly", s=28)
    ax.set_xlabel("DateTime")
    ax.set_ylabel("Vehicles / Traffic Volume")
    ax.set_title(f"Junction {int(selected_junction)} — Sequential Behavior vs GRU Prediction")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.info(
        f"An event is flagged when the absolute GRU prediction error exceeds "
        f"the {threshold_percentile:.1f}th-percentile threshold ({threshold:.2f} vehicles)."
    )

with tab2:
    if len(anomalies):
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
    else:
        st.success("No anomalies were detected at the current threshold.")

with tab3:
    if len(result):
        idx = st.number_input(
            "Select test event index",
            min_value=0, max_value=len(result)-1, value=0, step=1
        )
        row = result.iloc[int(idx)]
        a, b, c = st.columns(3)
        a.metric("Actual", f"{row['Actual']:.2f}")
        b.metric("GRU prediction", f"{row['Predicted']:.2f}")
        c.metric("Absolute error", f"{row['Absolute_Error']:.2f}")
        if bool(row["Anomaly"]):
            st.error("🚨 ANOMALY: sequential traffic behavior differs strongly from the GRU expectation.")
        else:
            st.success("✅ NORMAL: behavior is within the learned residual range.")

with tab4:
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
    "150 → 150 → 50 → 50 → 50 GRU units with 0.2 dropout and a Dense(1) output."
)
