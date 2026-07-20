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

/* Main app spacing */
.block-container{
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header hero */
.project-hero{
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #2563EB 100%);
    border-radius: 22px;
    padding: 28px 32px;
    color: #FFFFFF;
    margin-bottom: 20px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
}
.project-hero h1{
    margin: 0;
    font-size: 36px;
    font-weight: 850;
    letter-spacing: -0.5px;
}
.project-hero p{
    margin: 8px 0 0 0;
    font-size: 17px;
    color: #DBEAFE;
}

/* Metric cards - readable in both Light Mode and Dark Mode */
[data-testid="stMetric"]{
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid #CBD5E1;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 7px 20px rgba(15, 23, 42, 0.10);
}

[data-testid="stMetricLabel"]{
    font-size: 16px;
    font-weight: 750;
    color: #0F172A !important;
}

[data-testid="stMetricValue"]{
    font-size: 34px;
    font-weight: 850;
    color: #1D4ED8 !important;
}

[data-testid="stMetricDelta"]{
    color: #475569 !important;
}

/* Custom cards */
.info-card{
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 18px;
    padding: 18px 20px;
    margin: 10px 0 16px 0;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    color: #0F172A;
}
.info-card h3{
    margin: 0 0 8px 0;
    color: #0F172A;
}
.info-card p{
    margin: 0;
    color: #334155;
    font-size: 16px;
}
.status-card-good{
    background: linear-gradient(135deg, #DCFCE7 0%, #F0FDF4 100%);
    border: 1px solid #86EFAC;
    border-radius: 18px;
    padding: 22px;
    color: #14532D;
    box-shadow: 0 7px 20px rgba(20, 83, 45, 0.10);
}
.status-card-unhealthy{
    background: linear-gradient(135deg, #FEF3C7 0%, #FFFBEB 100%);
    border: 1px solid #FCD34D;
    border-radius: 18px;
    padding: 22px;
    color: #78350F;
    box-shadow: 0 7px 20px rgba(120, 53, 15, 0.10);
}
.status-card-hazardous{
    background: linear-gradient(135deg, #FEE2E2 0%, #FFF1F2 100%);
    border: 1px solid #FCA5A5;
    border-radius: 18px;
    padding: 22px;
    color: #7F1D1D;
    box-shadow: 0 7px 20px rgba(127, 29, 29, 0.10);
}
.status-card-good h2,
.status-card-unhealthy h2,
.status-card-hazardous h2{
    margin: 0 0 10px 0;
    font-size: 28px;
    font-weight: 850;
}
.status-card-good p,
.status-card-unhealthy p,
.status-card-hazardous p{
    margin: 0;
    font-size: 16px;
    font-weight: 600;
}
.small-muted{
    color: #64748B;
    font-size: 14px;
}

/* Dataframe/table */
[data-testid="stDataFrame"]{
    border-radius: 12px;
}

/* Sidebar text */
.sidebar-card{
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 14px;
    padding: 14px;
    color: #0F172A;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# AUTO REFRESH
# ==========================================

st_autorefresh(
    interval=4000,
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


def format_time_gap(seconds):
    """Convert seconds into a simple readable duration."""
    seconds = max(0, int(seconds))

    if seconds < 60:
        return f"{seconds} seconds"

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} minutes"

    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} hours"

    days = hours / 24
    return f"{days:.1f} days"


def get_data_upload_status(latest_time, timeout_seconds=120):
    """Check whether the latest uploaded data is still recent.

    The ESP32 uploads data every 10 seconds. A timeout of 120 seconds is used
    to avoid false warnings caused by internet delay, Google Sheets delay, or
    dashboard caching.
    """
    latest_time = pd.to_datetime(latest_time, errors="coerce")

    if pd.isna(latest_time):
        return "ERROR", None, "Latest timestamp is invalid. Please check Google Sheets timestamp format."

    # Google Sheets timestamp is treated as Malaysia local time.
    if getattr(latest_time, "tzinfo", None) is not None:
        latest_time = latest_time.tz_convert("Asia/Kuala_Lumpur").tz_localize(None)

    current_time = pd.Timestamp.now(tz="Asia/Kuala_Lumpur").tz_localize(None)
    time_gap_seconds = (current_time - latest_time).total_seconds()

    # If timestamp appears ahead of dashboard time, it is usually a timezone/clock issue.
    if time_gap_seconds < -120:
        return (
            "WARNING",
            abs(time_gap_seconds),
            "Latest data timestamp appears to be ahead of the dashboard clock. Please check device time or timezone setting."
        )

    if time_gap_seconds <= timeout_seconds:
        return (
            "OK",
            time_gap_seconds,
            f"Data upload is active. Latest sensor data was received {format_time_gap(time_gap_seconds)} ago."
        )

    if time_gap_seconds <= 300:
        return (
            "WARNING",
            time_gap_seconds,
            f"No new sensor data received for {format_time_gap(time_gap_seconds)}. Please check ESP32 Wi-Fi connection or Google Sheets upload."
        )

    return (
        "ERROR",
        time_gap_seconds,
        f"No new sensor data received for {format_time_gap(time_gap_seconds)}. Please check ESP32 power supply, Wi-Fi connection, or Google Sheets upload."
    )


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
<div class="project-hero">
    <h1>🌍 Smart Air Quality Monitoring System</h1>
    <p>Real-Time Monitoring • Historical Analysis • AI-Based 1-Hour Prediction • Smart Ventilation Control</p>
</div>
""", unsafe_allow_html=True)

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

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="sidebar-card">
<b>Final Year Project</b><br>
AI-Enhanced IoT Indoor Air Quality Monitoring, Prediction, and Smart Ventilation Control System<br><br>
<b>Developed by</b><br>
Muhammad Haikal Abdul Halim<br><br>
<b>Supervisor</b><br>
Dr. Nur Nabila Mohamed<br><br>
<b>Institution</b><br>
Faculty of Electrical Engineering<br>
UiTM Shah Alam
</div>
""", unsafe_allow_html=True)

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

    st.subheader("🛠 System Health Monitoring")

    upload_status, gap_seconds, upload_message = get_data_upload_status(
        latest["Time"],
        timeout_seconds=30
    )

    if upload_status == "OK":
        st.success(f"✅ {upload_message}")
    elif upload_status == "WARNING":
        st.warning(f"⚠ {upload_message}")
    else:
        st.error(f"🚨 {upload_message}")

    st.caption(
        "This health check monitors whether new data is still being uploaded to Google Sheets. "
        "It does not confirm physical fan/servo movement because actuator feedback sensors are not installed."
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
        f"Charts are displayed using 1-minute averaged data ({len(df_chart):,} points) from {len(df):,} original records to improve dashboard performance while preserving the overall trend."
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

        st.markdown("""
        <div class="info-card">
            <h3>🤖 AI Prediction Module</h3>
            <p>This module predicts the indoor air quality condition <b>1 hour ahead</b> using historical sensor data. The model becomes more useful as more valid records are collected over time.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📘 Simple explanation of AI results", expanded=False):
            st.markdown("""
            **1-Hour Status Accuracy** shows how often the AI correctly predicts the air quality category after 1 hour, such as GOOD, UNHEALTHY, or HAZARDOUS.  
            **Mean Absolute Error (MAE)** shows the average difference between the predicted AQ and the actual AQ after 1 hour. Lower value is better.  
            **Root Mean Squared Error (RMSE)** gives higher penalty to larger prediction errors. Lower value is better.  

            The dashboard focuses on **status accuracy** because the system decision is based on air quality category and smart recommendation, not only the exact AQ value.
            """)

        col_a, col_b, col_c = st.columns(3)

        col_a.metric(
            "1-Hour Status Accuracy",
            f"{status_accuracy * 100:.2f}%"
        )

        col_b.metric(
            "Mean Absolute Error (MAE)",
            f"±{model_mae:.0f} AQ"
        )

        col_c.metric(
            "Root Mean Squared Error (RMSE)",
            f"±{model_rmse:.0f} AQ"
        )

        st.caption(
            f"AI-ready records: {ai_dataset_count:,}. Training records: {train_size_count:,}. Testing records: {test_size_count:,}. Evaluation uses chronological unseen data."
        )

        if baseline_mae is not None and baseline_mae > 0:
            improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
            if improvement >= 0:
                st.success(
                    f"The AI prediction error is {improvement:.2f}% lower than a simple method that assumes the AQ value will stay the same after 1 hour."
                )
            else:
                st.warning(
                    f"The AI prediction error is {-improvement:.2f}% higher than the simple baseline. More data or feature tuning may improve the model."
                )

        with st.expander("🔍 Advanced model details"):
            st.write(f"R² Score: {model_r2:.4f}")
            st.write(f"Baseline MAE: {baseline_mae:.2f} AQ")
            st.write("R² is kept only as a diagnostic value. For this project, status accuracy and prediction error are easier to interpret because the system mainly decides whether the future condition is GOOD, UNHEALTHY, or HAZARDOUS.")
            st.write("Note: Newly collected data can be used immediately for live prediction. However, for training and evaluation, a new row only becomes AI-ready after its actual 1-hour-ahead value exists.")

        # Build latest valid feature row for live prediction.
        feature_table = build_feature_table(df)
        latest_features_df = feature_table.dropna(subset=feature_cols).copy()

        if latest_features_df.empty:
            st.warning("Not enough recent data to generate lag-based AI prediction yet. Please allow the system to collect more continuous records.")
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
                "Predicted AQ in 1 Hour",
                int(predictedAQ)
            )

            col3.metric(
                "Predicted Status in 1 Hour",
                predictedStatus
            )

            if predictedStatus == "GOOD":
                st.markdown("""
                <div class="status-card-good">
                    <h2>🟢 Predicted Status: GOOD</h2>
                    <p>Air quality is expected to remain safe within the next hour. No immediate ventilation action is required.</p>
                </div>
                """, unsafe_allow_html=True)

            elif predictedStatus == "UNHEALTHY":
                st.markdown("""
                <div class="status-card-unhealthy">
                    <h2>🟠 Predicted Status: UNHEALTHY</h2>
                    <p>Air quality may deteriorate within the next hour. Ventilation is recommended.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="status-card-hazardous">
                    <h2>🔴 Predicted Status: HAZARDOUS</h2>
                    <p>Hazardous air quality is predicted. Immediate ventilation action is recommended.</p>
                </div>
                """, unsafe_allow_html=True)

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
                ⚠ Air quality is expected to deteriorate

                • Open windows
                • Turn ON exhaust fan
                • Increase ventilation
                • Reduce pollution source if possible
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
