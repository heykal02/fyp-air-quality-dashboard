import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from streamlit_autorefresh import st_autorefresh

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Smart Air Quality Monitoring System",
    page_icon="🌍",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

[data-testid="stMetric"]{
    background-color:#111827;
    border:1px solid #374151;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

[data-testid="stMetricLabel"]{
    font-size:18px;
    font-weight:bold;
}

[data-testid="stMetricValue"]{
    font-size:35px;
    color:#60A5FA;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# AUTO REFRESH
# ==========================================

st_autorefresh(
    interval=10000,
    key="air_quality_refresh"
)

# ==========================================
# GOOGLE SHEET CSV
# ==========================================

GOOGLE_SHEET_CSV = "https://docs.google.com/spreadsheets/d/1KjA7kCINJ9EWEUoDvc6GVX7-19r2hzJhe6COvuESbx0/export?format=csv"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def classify_aq(aq_value):
    """Classify AQ value using project thresholds."""
    if aq_value <= 3300:
        return "GOOD"
    elif aq_value <= 3900:
        return "UNHEALTHY"
    return "HAZARDOUS"


def add_valid_lag(df, source_col, lag_steps, lag_seconds, output_col):
    """Add lag feature only when timestamps are truly separated by the intended duration.

    This prevents wrong lag values caused by daily gaps or system downtime.
    """
    shifted_values = df[source_col].shift(lag_steps)
    shifted_time = df["Time"].shift(lag_steps)
    time_diff = (df["Time"] - shifted_time).dt.total_seconds()
    valid_lag = time_diff.between(lag_seconds - 20, lag_seconds + 20)
    df[output_col] = shifted_values.where(valid_lag)
    return df


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data(ttl=30)
def load_data():

    df = pd.read_csv(
        GOOGLE_SHEET_CSV,
        skiprows=1
    )

    df.columns = [
        "Time",
        "Air Quality",
        "Temperature",
        "Humidity",
        "Status"
    ]

    df = df.dropna()

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    df["Air Quality"] = pd.to_numeric(
        df["Air Quality"],
        errors="coerce"
    )

    df["Temperature"] = pd.to_numeric(
        df["Temperature"],
        errors="coerce"
    )

    df["Humidity"] = pd.to_numeric(
        df["Humidity"],
        errors="coerce"
    )

    df = df.dropna()
    df = df.sort_values("Time").drop_duplicates("Time").reset_index(drop=True)

    return df


# ==========================================
# DOWNSAMPLE CHART DATA
# ==========================================

@st.cache_data(ttl=30)
def load_chart_data():
    df = load_data()
    df_chart = (
        df.set_index("Time")
        .resample("1min")
        .agg({
            "Air Quality": "mean",
            "Temperature": "mean",
            "Humidity": "mean"
        })
        .dropna()
        .reset_index()
    )
    return df_chart


# ==========================================
# AI FEATURE ENGINEERING
# ==========================================

def build_feature_table(df):
    """Create feature table for 1-hour-ahead AQ prediction.

    Improvements from the previous version:
    1. Uses time-based features because daily spikes happen at repeated time windows.
    2. Uses lag and rolling features to capture recent AQ movement.
    3. Handles daily operating gaps and downtime so the model does not learn wrong 1-hour targets.
    """
    df_feat = df.copy().sort_values("Time").reset_index(drop=True)

    # Time-of-day features help the model learn recurring daily patterns.
    df_feat["Hour"] = df_feat["Time"].dt.hour
    df_feat["Minute"] = df_feat["Time"].dt.minute
    df_feat["DayOfWeek"] = df_feat["Time"].dt.dayofweek
    df_feat["MinuteOfDay"] = df_feat["Hour"] * 60 + df_feat["Minute"]
    df_feat["Time_Sin"] = np.sin(2 * np.pi * df_feat["MinuteOfDay"] / 1440)
    df_feat["Time_Cos"] = np.cos(2 * np.pi * df_feat["MinuteOfDay"] / 1440)

    # Current AQ status encoded as an additional non-leaking feature.
    status_map = {
        "GOOD": 0,
        "UNHEALTHY": 1,
        "HAZARDOUS": 2
    }
    df_feat["Current_Status_Code"] = df_feat["Air Quality"].apply(classify_aq).map(status_map)

    # Lag features based on 10-second sampling interval.
    # 6 samples = 1 minute, 30 = 5 minutes, 90 = 15 minutes, 180 = 30 minutes.
    lag_config = [
        ("1min", 6, 60),
        ("5min", 30, 300),
        ("15min", 90, 900),
        ("30min", 180, 1800),
    ]

    for label, steps, seconds in lag_config:
        df_feat = add_valid_lag(df_feat, "Air Quality", steps, seconds, f"AQ_Lag_{label}")

    # Temperature and humidity recent context.
    df_feat = add_valid_lag(df_feat, "Temperature", 30, 300, "Temp_Lag_5min")
    df_feat = add_valid_lag(df_feat, "Humidity", 30, 300, "Hum_Lag_5min")

    # Rolling features using timestamp windows.
    df_indexed = df_feat.set_index("Time")
    for window in ["5min", "15min", "30min"]:
        df_feat[f"AQ_Roll_Mean_{window}"] = (
            df_indexed["Air Quality"].rolling(window, min_periods=1).mean().values
        )
        df_feat[f"AQ_Roll_Std_{window}"] = (
            df_indexed["Air Quality"].rolling(window, min_periods=2).std().fillna(0).values
        )

    df_feat["Temp_Roll_Mean_15min"] = (
        df_indexed["Temperature"].rolling("15min", min_periods=1).mean().values
    )
    df_feat["Hum_Roll_Mean_15min"] = (
        df_indexed["Humidity"].rolling("15min", min_periods=1).mean().values
    )

    return df_feat


@st.cache_resource(ttl=300)
def train_model():

    df = load_data()
    df_feat = build_feature_table(df)

    # Generate true 1-hour-ahead target using timestamp matching instead of simple row shifting.
    # This avoids wrong targets across midnight gaps and maintenance downtime.
    left = df_feat.copy()
    left["Target_Time"] = left["Time"] + pd.Timedelta(hours=1)

    right = df[["Time", "Air Quality"]].copy()
    right = right.rename(columns={
        "Time": "Future_Time",
        "Air Quality": "Future_AQ"
    })

    df_ai = pd.merge_asof(
        left.sort_values("Target_Time"),
        right.sort_values("Future_Time"),
        left_on="Target_Time",
        right_on="Future_Time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=20)
    )

    df_ai = df_ai.sort_values("Time").reset_index(drop=True)
    df_ai["Future_Status"] = df_ai["Future_AQ"].apply(lambda x: classify_aq(x) if pd.notna(x) else np.nan)

    feature_cols = [
        "Air Quality",
        "Temperature",
        "Humidity",
        "Current_Status_Code",
        "Hour",
        "MinuteOfDay",
        "DayOfWeek",
        "Time_Sin",
        "Time_Cos",
        "AQ_Lag_1min",
        "AQ_Lag_5min",
        "AQ_Lag_15min",
        "AQ_Lag_30min",
        "Temp_Lag_5min",
        "Hum_Lag_5min",
        "AQ_Roll_Mean_5min",
        "AQ_Roll_Std_5min",
        "AQ_Roll_Mean_15min",
        "AQ_Roll_Std_15min",
        "AQ_Roll_Mean_30min",
        "AQ_Roll_Std_30min",
        "Temp_Roll_Mean_15min",
        "Hum_Roll_Mean_15min",
    ]

    df_ai = df_ai.dropna(subset=feature_cols + ["Future_AQ", "Future_Status"]).copy()

    if len(df_ai) <= 1500:
        return None, None, False, None, None, None, None, None, None, None, feature_cols

    X = df_ai[feature_cols]
    y = df_ai["Future_AQ"]

    # Chronological split prevents future data leakage.
    split_index = int(len(df_ai) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=22,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    actual_status = [classify_aq(v) for v in y_test]
    predicted_status = [classify_aq(v) for v in y_pred]
    status_accuracy = accuracy_score(actual_status, predicted_status)

    # Baseline: assume AQ after 1 hour remains the same as current AQ.
    baseline_pred = X_test["Air Quality"].values
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    train_size_count = len(X_train)
    test_size_count = len(X_test)
    model_ready = True

    return (
        model,
        r2,
        model_ready,
        train_size_count,
        test_size_count,
        mae,
        rmse,
        status_accuracy,
        baseline_mae,
        len(df_ai),
        feature_cols
    )


# ==========================================
# LOAD ALL DATA AND AI MODEL
# ==========================================

df = load_data()
df_chart = load_chart_data()
(
    model,
    model_r2,
    model_ready,
    train_size_count,
    test_size_count,
    model_mae,
    model_rmse,
    status_accuracy,
    baseline_mae,
    ai_dataset_count,
    feature_cols
) = train_model()

# ==========================================
# HEADER
# ==========================================

st.markdown("""
# 🌍 Smart Air Quality Monitoring System

### Real-Time Monitoring and AI-Based Prediction Dashboard

---
""")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🌍 Air Quality System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Realtime Monitoring",
        "Historical Analysis",
        "Trend Analysis",
        "AI Prediction"
    ]
)
st.sidebar.markdown("---")

csv = df.to_csv(index=False)

st.sidebar.download_button(
    "📥 Download Dataset",
    csv,
    "air_quality_dataset.csv",
    "text/csv"
)

st.sidebar.markdown("---")

st.sidebar.info(
    f"Total Records: {len(df):,}"
)

# ==========================================
# REALTIME MONITORING
# ==========================================

if page == "Realtime Monitoring":

    latest = df.iloc[-1]

    st.subheader("📡 Realtime Monitoring")

    status = latest["Status"]

    if status == "GOOD":
        st.success("🟢 Current Air Quality Status : GOOD")

    elif status == "UNHEALTHY":
        st.warning("🟠 Current Air Quality Status : UNHEALTHY")

    else:
        st.error("🔴 Current Air Quality Status : HAZARDOUS")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🌫 Air Quality",
        int(latest["Air Quality"])
    )

    col2.metric(
        "🌡 Temperature (°C)",
        round(latest["Temperature"], 1)
    )

    col3.metric(
        "💧 Humidity (%)",
        round(latest["Humidity"], 1)
    )

    col4.metric(
        "📊 Status",
        latest["Status"]
    )

    st.markdown("---")

    st.subheader("📡 Air Quality Gauge")

    aq = int(latest["Air Quality"])

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aq,
            title={"text": "Air Quality"},
            gauge={
                "axis": {"range": [0, 4500]},
                "bgcolor": "rgba(0,0,0,0)",
                "bar": {"color": "#111827", "thickness": 0},
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 3300],
                        "color": "green",
                        "line": {"color": "green", "width": 0}
                    },
                    {
                        "range": [3300, 3900],
                        "color": "orange",
                        "line": {"color": "orange", "width": 0}
                    },
                    {
                        "range": [3900, 4500],
                        "color": "red",
                        "line": {"color": "red", "width": 0}
                    }
                ]
            }
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        f"🕒 Last Update : {latest['Time']}"
    )

    st.markdown("---")

    st.subheader("📋 Latest Sensor Records")

    st.dataframe(
        df.tail(20),
        use_container_width=True
    )

# ==========================================
# HISTORICAL ANALYSIS
# ==========================================

elif page == "Historical Analysis":

    st.subheader("📈 Historical Analysis")

    st.caption(
        f"Chart dipapar menggunakan data 1-minit average ({len(df_chart):,} points) daripada {len(df):,} rekod asal untuk prestasi lebih laju."
    )

    fig1 = px.line(
        df_chart,
        x="Time",
        y="Air Quality",
        title="Air Quality Trend",
        template="plotly_white"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.line(
        df_chart,
        x="Time",
        y="Temperature",
        title="Temperature Trend",
        template="plotly_white"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3 = px.line(
        df_chart,
        x="Time",
        y="Humidity",
        title="Humidity Trend",
        template="plotly_white"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# ==========================================
# TREND ANALYSIS
# ==========================================

elif page == "Trend Analysis":

    st.subheader("📊 Trend Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average AQ",
        round(df["Air Quality"].mean(), 2)
    )

    col2.metric(
        "Maximum AQ",
        int(df["Air Quality"].max())
    )

    col3.metric(
        "Minimum AQ",
        int(df["Air Quality"].min())
    )

    st.markdown("---")

    st.subheader("📈 Statistical Summary")

    stats = df[["Air Quality", "Temperature", "Humidity"]].agg(["mean", "std", "min", "max"])
    stats.index = ["Mean", "Std Dev", "Min", "Max"]

    st.dataframe(
        stats.round(2),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🥧 Status Distribution")

    status_count = df["Status"].value_counts()

    fig_pie = px.pie(
        values=status_count.values,
        names=status_count.index,
        title="Status Distribution"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🔥 Correlation Heatmap")

    corr = df[
        [
            "Air Quality",
            "Temperature",
            "Humidity"
        ]
    ].corr()

    annotation_text = corr.round(2).astype(str).values

    fig_heatmap = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=annotation_text
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )

# ==========================================
# AI PREDICTION
# ==========================================

elif page == "AI Prediction":

    st.subheader("🤖 AI Prediction")

    if model_ready:

        st.info(
            "Machine Learning model predicts indoor air quality 1 hour ahead using current sensor readings, time-of-day patterns, lag features, and rolling average features."
        )

        col_a, col_b, col_c = st.columns(3)

        col_a.metric(
            "1-Hour Status Accuracy",
            f"{status_accuracy * 100:.2f}%"
        )

        col_b.metric(
            "MAE",
            f"{model_mae:.2f} AQ units"
        )

        col_c.metric(
            "RMSE",
            f"{model_rmse:.2f} AQ units"
        )

        st.caption(
            f"AI-ready records: {ai_dataset_count:,}. Trained on {train_size_count:,} records and tested on {test_size_count:,} unseen chronological records."
        )

        if baseline_mae is not None and baseline_mae > 0:
            improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
            if improvement >= 0:
                st.success(
                    f"Model MAE improved by {improvement:.2f}% compared with a simple persistence baseline."
                )
            else:
                st.warning(
                    f"Model MAE is {-improvement:.2f}% higher than the simple persistence baseline. More tuning may be required."
                )

        with st.expander("Show model diagnostics"):
            st.write(f"R² Score: {model_r2:.4f}")
            st.write(f"Baseline MAE: {baseline_mae:.2f} AQ units")
            st.write("Note: R² can become negative for noisy sensor data or when future AQ spikes are difficult to predict exactly. Therefore, status accuracy and MAE are displayed as the main dashboard metrics.")

        # Build latest valid feature row for live prediction.
        feature_table = build_feature_table(df)
        latest_features_df = feature_table.dropna(subset=feature_cols).copy()

        if latest_features_df.empty:
            st.warning("Not enough recent data to generate lag-based AI prediction yet.")
        else:
            latest_feature_row = latest_features_df.iloc[-1]
            latest_actual_row = df.iloc[-1]

            latest_X = pd.DataFrame(
                [latest_feature_row[feature_cols].values],
                columns=feature_cols
            )

            predictedAQ = float(model.predict(latest_X)[0])
            predictedStatus = classify_aq(predictedAQ)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Current AQ",
                int(latest_actual_row["Air Quality"])
            )

            col2.metric(
                "Predicted AQ After 1 Hour",
                int(predictedAQ)
            )

            col3.metric(
                "Predicted Status",
                predictedStatus
            )

            if predictedStatus == "GOOD":

                st.success(
                    f"🟢 Predicted Status: {predictedStatus}"
                )

                st.success(
                    "Air quality is expected to remain safe within the next hour."
                )

            elif predictedStatus == "UNHEALTHY":

                st.warning(
                    f"🟠 Predicted Status: {predictedStatus}"
                )

                st.warning(
                    "Ventilation is recommended for the next hour."
                )

            else:

                st.error(
                    f"🔴 Predicted Status: {predictedStatus}"
                )

                st.error(
                    "Hazardous air quality predicted. Open windows and activate exhaust fan."
                )

            st.markdown("---")

            st.subheader("🧠 Smart Recommendation")

            if predictedStatus == "GOOD":

                st.success("""
                ✅ Air quality is healthy

                • No action required
                • Ventilation is sufficient
                • Continue monitoring
                """)

            elif predictedStatus == "UNHEALTHY":

                st.warning("""
                ⚠ Air quality is deteriorating

                • Open windows
                • Turn ON exhaust fan
                • Increase ventilation
                • Reduce occupancy
                """)

            else:

                st.error("""
                🚨 Hazardous air quality predicted

                • Open all windows immediately
                • Activate exhaust fan
                • Reduce room occupancy
                • Investigate pollution source
                """)

    else:

        st.warning(
            "Not enough data available for AI prediction."
        )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Developed by Muhammad Haikal Abdul Halim | Bachelor of Electrical & Electronics Engineering Technology (Hons) | UiTM Shah Alam"
)
