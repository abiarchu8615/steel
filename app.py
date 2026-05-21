import json
import os
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports"

st.set_page_config(page_title="AI Digital Twin for Steel Manufacturing", layout="wide")
st.title("AI Digital Twin System for Steel Manufacturing Predictive Maintenance")
st.write("Predict machine failure, detect degradation trends, generate maintenance tickets, and send Telegram/email alerts.")

selected_page = st.sidebar.radio(
    "Choose Module",
    [
        "Overview",
        "Machine Health",
        "Failure Prediction Demo",
        "Failure Trend Prediction",
        "AI Chatbot",
        "Maintenance Tickets",
    ],
)

@st.cache_data
def load_ai4i():
    path = DATA / "ai4i2020.csv"
    if not path.exists():
        st.error(f"Missing dataset: {path}")
        st.stop()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Sample_Index"] = np.arange(len(df))
    df["Temperature difference [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Mechanical power proxy"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["Failure Type"] = df.apply(get_failure_type, axis=1)
    return df

def get_failure_type(row):
    labels = []
    mapping = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure",
    }
    for col, name in mapping.items():
        if int(row.get(col, 0)) == 1:
            labels.append(name)
    return ", ".join(labels) if labels else "Normal"

def load_metrics():
    path = REPORTS / "machine_failure_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def render_model_evaluation(model, metrics):
    """Show ML evaluation metrics required for the predictive maintenance report."""
    if not metrics:
        st.info("Model metrics not found yet. Retrain the model to generate the evaluation report.")
        return

    st.subheader("Model Performance Evaluation")

    report = metrics.get("classification_report", {})
    failure_report = report.get("1", {}) or report.get(1, {})

    accuracy = float(metrics.get("accuracy", 0))
    precision = float(failure_report.get("precision", 0))
    recall = float(failure_report.get("recall", 0))
    f1 = float(failure_report.get("f1-score", 0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")

    st.caption(
        "Precision, Recall, and F1-score are calculated for the failure class, "
        "because detecting failures is more important than only predicting normal machines."
    )

    if "confusion_matrix" in metrics:
        st.write("### Confusion Matrix")
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual Normal", "Actual Failure"],
            columns=["Predicted Normal", "Predicted Failure"],
        )
        st.dataframe(cm, use_container_width=True)

        cm_long = cm.reset_index().melt(id_vars="index", var_name="Prediction", value_name="Count")
        cm_long = cm_long.rename(columns={"index": "Actual"})
        st.plotly_chart(
            px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix Heatmap",
            ),
            use_container_width=True,
        )

    st.write("### Feature Importance")
    feature_importance = metrics.get("feature_importance", [])
    if feature_importance:
        importance_df = pd.DataFrame(feature_importance)
    else:
        # Fallback for older metric files: derive names from the fitted pipeline if available.
        try:
            preprocess = model.named_steps["preprocess"]
            classifier = model.named_steps["classifier"]
            feature_names = preprocess.get_feature_names_out()
            importance_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Importance": classifier.feature_importances_,
                }
            )
        except Exception:
            importance_df = pd.DataFrame()

    if not importance_df.empty:
        importance_df["Feature"] = (
            importance_df["Feature"]
            .astype(str)
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
        )
        importance_df = importance_df.sort_values("Importance", ascending=False).head(10)
        st.plotly_chart(
            px.bar(
                importance_df.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Features Influencing Machine Failure",
            ),
            use_container_width=True,
        )
    else:
        st.info("Feature importance is unavailable for this model file.")

    with st.expander("View detailed classification report"):
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

def classify_maintenance_risk(probability):
    """Convert failure probability into an easy-to-understand risk level."""
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.40:
        return "MEDIUM"
    return "LOW"

# =========================
# ALERT HELPERS
# =========================
def send_telegram_alert(message):

    bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "").strip()

    if not bot_token or not chat_id:
        st.error("Telegram secrets missing.")
        return False

    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        payload = {
            "chat_id": chat_id,
            "text": str(message),
        }

        response = requests.post(url, json=payload, timeout=10)
        result = response.json()

       

        if response.status_code == 200 and result.get("ok"):
            return True

        st.error(result)
        return False

    except Exception as e:
        st.error(f"Telegram alert failed: {e}")
        return False




def send_email_alert(subject, message):
    sender_email = st.secrets.get("EMAIL_SENDER", os.getenv("EMAIL_SENDER", "")).strip()
    sender_password = st.secrets.get("EMAIL_PASSWORD", os.getenv("EMAIL_PASSWORD", "")).strip()
    receiver_email = st.secrets.get("EMAIL_RECEIVER", os.getenv("EMAIL_RECEIVER", "")).strip()

    if not all([sender_email, sender_password, receiver_email]):
        st.warning("Email credentials not configured. Set EMAIL_SENDER, EMAIL_PASSWORD, and EMAIL_RECEIVER.")
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg.set_content(str(message))

        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email alert failed: {e}")
        return False

def build_alert_message(ticket):
    return f"""STEEL MANUFACTURING MACHINE ALERT

Ticket ID: {ticket['ticket_id']}
Priority: {ticket['priority']}
Module: {ticket['module']}
Signal: {ticket['asset_or_signal']}
Failure Status: {ticket['failure_status']}
Due Time: {ticket['due_time']}
Risk Probability: {ticket['risk_probability']}
Severity Score: {ticket['severity_score']}

Likely Cause:
{ticket['likely_cause']}

Recommended Action:
{ticket['maintenance_action']}

Owner:
{ticket['owner']}
"""

def render_alert_buttons(ticket):
    st.subheader("Send Alert / Push Notification")
    alert_message = build_alert_message(ticket)
    st.text_area("Alert Message Preview", alert_message, height=260)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Send Telegram Alert", key=f"telegram_{ticket['ticket_id']}"):
            if send_telegram_alert(alert_message):
                st.success("Telegram alert sent successfully.")
            else:
                st.error("Telegram alert failed.")
    with col2:
        if st.button("Send Email Alert", key=f"email_{ticket['ticket_id']}"):
            if send_email_alert(f"Steel Machine Alert - {ticket['ticket_id']}", alert_message):
                st.success("Email alert sent successfully.")
            else:
                st.error("Email alert failed.")
                
def auto_send_alert(ticket):
    sent_key = f"alert_sent_{ticket['ticket_id']}"

    if st.session_state.get(sent_key):
        return

    message = build_alert_message(ticket)

    if send_telegram_alert(message):
        st.success("Automatic Telegram alert sent.")

    st.session_state[sent_key] = True

# =========================
# FAILURE TREND ENGINE
# =========================
def detect_failure_trend(df, time_col, value_col, warning_threshold=None, failure_threshold=None, direction="above", window=10):
    work = df.copy()
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[time_col, value_col]).sort_values(time_col).reset_index(drop=True)
    if work.empty:
        return work

    window = max(2, min(int(window), len(work)))
    work["rolling_mean"] = work[value_col].rolling(window=window, min_periods=1).mean()
    work["rolling_std"] = work[value_col].rolling(window=window, min_periods=1).std().fillna(0)
    work["trend_change"] = work["rolling_mean"].diff().fillna(0)

    if warning_threshold is None:
        warning_threshold = work[value_col].quantile(0.80 if direction == "above" else 0.20)
    if failure_threshold is None:
        failure_threshold = work[value_col].quantile(0.95 if direction == "above" else 0.05)

    if direction == "above":
        work["warning_zone"] = work["rolling_mean"] >= warning_threshold
        work["failure_zone"] = work["rolling_mean"] >= failure_threshold
        work["moving_towards_failure"] = work["trend_change"] > 0
    else:
        work["warning_zone"] = work["rolling_mean"] <= warning_threshold
        work["failure_zone"] = work["rolling_mean"] <= failure_threshold
        work["moving_towards_failure"] = work["trend_change"] < 0

    work["early_warning"] = work["warning_zone"] & work["moving_towards_failure"] & ~work["failure_zone"]
    work["failure_detected"] = work["failure_zone"]

    def status(row):
        if row["failure_detected"]:
            return "FAILURE"
        if row["early_warning"]:
            return "EARLY WARNING"
        if row["warning_zone"]:
            return "WATCH"
        return "NORMAL"

    work["failure_status"] = work.apply(status, axis=1)
    return add_failure_severity_score(work)

def add_failure_severity_score(trend_df):
    work = trend_df.copy()
    if work.empty:
        return work
    max_abs_trend = work["trend_change"].abs().max()
    max_std = work["rolling_std"].max()
    normalized_trend = 0 if max_abs_trend == 0 else work["trend_change"].abs() / max_abs_trend
    normalized_std = 0 if max_std == 0 else work["rolling_std"] / max_std
    status_probability = {"NORMAL": 0.10, "WATCH": 0.40, "EARLY WARNING": 0.70, "FAILURE": 1.00}
    work["risk_probability"] = work["failure_status"].map(status_probability).fillna(0.10)
    work["severity_score"] = (normalized_trend * normalized_std * work["risk_probability"]).fillna(0)
    work["severity_level"] = work["severity_score"].apply(classify_severity)
    return work

def classify_severity(score):
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.50:
        return "HIGH"
    if score >= 0.25:
        return "MEDIUM"
    return "LOW"

def forecast_future_trend(trend_df, time_col, value_col, periods=15):
    work = trend_df.dropna(subset=[time_col, value_col]).copy()
    if len(work) < 3:
        return pd.DataFrame()
    recent = work.tail(min(30, len(work))).copy()
    recent["x"] = range(len(recent))
    slope, intercept = np.polyfit(recent["x"], recent[value_col], 1)
    last_x = recent["x"].iloc[-1]
    last_time = work[time_col].iloc[-1]
    rows = []
    for i in range(1, periods + 1):
        rows.append({time_col: last_time + i, f"forecast_{value_col}": slope * (last_x + i) + intercept, "forecast_step": i})
    return pd.DataFrame(rows)

# =========================
# STEEL MAINTENANCE LOGIC
# =========================
def get_signal_map():
    return {
        "Motor / Drive System": [("Rotational speed [rpm]", "below"), ("Torque [Nm]", "above"), ("Mechanical power proxy", "above")],
        "Thermal System": [("Air temperature [K]", "above"), ("Process temperature [K]", "above"), ("Temperature difference [K]", "above")],
        "Tool Wear System": [("Tool wear [min]", "above"), ("Torque [Nm]", "above")],
        "Overall Machine": [("Machine failure", "above"), ("Tool wear [min]", "above"), ("Mechanical power proxy", "above")],
    }

def get_prescriptive_action(module, signal, status):
    if status == "NORMAL":
        return {
            "priority": "LOW",
            "timeframe": "Routine monitoring",
            "owner": "Operations Team",
            "likely_cause": "No abnormal degradation pattern detected.",
            "action": "Continue normal monitoring and scheduled preventive maintenance.",
        }

    priority = {"WATCH": "MEDIUM", "EARLY WARNING": "HIGH", "FAILURE": "CRITICAL"}.get(status, "LOW")
    timeframe = {"WATCH": "Inspect within 7 days", "EARLY WARNING": "Inspect within 24-48 hours", "FAILURE": "Immediate intervention required"}.get(status, "Routine")

    if signal in ["Tool wear [min]"]:
        cause = "Tool wear is increasing and may lead to cutting/tool failure or poor product quality."
        action = "Inspect tool condition, check wear limit, plan tool replacement, and verify product surface quality."
        owner = "Mechanical Maintenance / Production Team"
    elif signal in ["Torque [Nm]", "Mechanical power proxy"]:
        cause = "Possible overload, material jam, drive stress, bearing friction, or abnormal process load."
        action = "Check drive load, inspect gearbox/bearings, verify material feed, and reduce load if abnormal torque continues."
        owner = "Mechanical + Electrical Maintenance Team"
    elif signal in ["Rotational speed [rpm]"]:
        cause = "Possible motor speed instability, drive fault, belt/coupling issue, or load imbalance."
        action = "Inspect motor, VFD/drive settings, coupling/belt condition, and verify speed feedback sensor."
        owner = "Electrical Maintenance Team"
    elif signal in ["Air temperature [K]", "Process temperature [K]", "Temperature difference [K]"]:
        cause = "Thermal condition is moving toward unsafe range, possibly due to cooling inefficiency or heat dissipation failure."
        action = "Check cooling airflow, heat exchanger, lubrication/cooling system, and temperature sensor calibration."
        owner = "Process / Utility Maintenance Team"
    elif signal == "Machine failure":
        cause = "Machine failure label or failure-risk indicator is active."
        action = "Stop or slow machine if required, inspect failure type indicators, and assign maintenance team immediately."
        owner = "Maintenance Supervisor"
    else:
        cause = "Machine signal is moving toward abnormal condition."
        action = "Inspect related machine components and validate sensor readings."
        owner = "Maintenance Team"

    return {"priority": priority, "timeframe": timeframe, "owner": owner, "likely_cause": cause, "action": action}

def calculate_due_time(priority):
    now = datetime.now()
    if priority == "CRITICAL":
        return now + timedelta(hours=2)
    if priority == "HIGH":
        return now + timedelta(hours=24)
    if priority == "MEDIUM":
        return now + timedelta(days=3)
    return now + timedelta(days=7)

def generate_ticket_id(module, signal):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    module_code = "".join(word[0] for word in module.split() if word[0].isalnum()).upper()
    signal_code = "".join(ch for ch in signal if ch.isalnum())[:6].upper()
    return f"ST-{module_code}-{signal_code}-{timestamp}"

def generate_ticket(module, signal, latest_row, forecast_warning="Not forecasted", forecast_failure="Not forecasted"):
    status = latest_row.get("failure_status", "NORMAL")
    severity = latest_row.get("severity_level", "LOW")
    score = float(latest_row.get("severity_score", 0))
    risk = float(latest_row.get("risk_probability", 0.10))
    action_plan = get_prescriptive_action(module, signal, status)
    priority = severity if severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"] else action_plan["priority"]

    return {
        "ticket_id": generate_ticket_id(module, signal),
        "module": module,
        "asset_or_signal": signal,
        "failure_status": status,
        "priority": priority,
        "due_time": calculate_due_time(priority).strftime("%Y-%m-%d %H:%M"),
        "risk_probability": f"{risk:.0%}",
        "severity_score": round(score, 3),
        "forecast_warning_time": str(forecast_warning),
        "forecast_failure_time": str(forecast_failure),
        "maintenance_action": action_plan["action"],
        "likely_cause": action_plan["likely_cause"],
        "owner": action_plan["owner"],
        "timeframe": action_plan["timeframe"],
        "status": "OPEN",
    }

def show_ticket(ticket):
    st.subheader("Autonomous Maintenance Ticket")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticket ID", ticket["ticket_id"])
    c2.metric("Priority", ticket["priority"])
    c3.metric("Due Time", ticket["due_time"])
    c4.metric("Risk", ticket["risk_probability"])

    if ticket["priority"] == "CRITICAL":
        st.error(ticket["maintenance_action"])
    elif ticket["priority"] == "HIGH":
        st.warning(ticket["maintenance_action"])
    elif ticket["priority"] == "MEDIUM":
        st.info(ticket["maintenance_action"])
    else:
        st.success(ticket["maintenance_action"])
    st.write("**Likely Cause:**", ticket["likely_cause"])
    st.write("**Owner:**", ticket["owner"])
    st.write("**Forecast Warning Time:**", ticket["forecast_warning_time"])
    st.write("**Forecast Failure Time:**", ticket["forecast_failure_time"])

    st.download_button(
        "Download Maintenance Ticket CSV",
        pd.DataFrame([ticket]).to_csv(index=False),
        f"{ticket['ticket_id']}.csv",
        "text/csv"
    )

    auto_send_alert(ticket)
    render_alert_buttons(ticket)

def get_latest_trend_ticket(module, signal, direction, window=10, forecast_periods=15):
    df = load_ai4i()
    warning_threshold = df[signal].quantile(0.80 if direction == "above" else 0.20)
    failure_threshold = df[signal].quantile(0.95 if direction == "above" else 0.05)
    trend_df = detect_failure_trend(df, "Sample_Index", signal, warning_threshold, failure_threshold, direction, window)
    forecast_df = forecast_future_trend(trend_df, "Sample_Index", signal, forecast_periods)

    forecast_warning = "Not forecasted"
    forecast_failure = "Not forecasted"
    if not forecast_df.empty:
        fcol = f"forecast_{signal}"
        if direction == "above":
            forecast_df["forecast_warning"] = forecast_df[fcol] >= warning_threshold
            forecast_df["forecast_failure"] = forecast_df[fcol] >= failure_threshold
        else:
            forecast_df["forecast_warning"] = forecast_df[fcol] <= warning_threshold
            forecast_df["forecast_failure"] = forecast_df[fcol] <= failure_threshold
        fw = forecast_df[forecast_df["forecast_warning"]]["Sample_Index"].min()
        ff = forecast_df[forecast_df["forecast_failure"]]["Sample_Index"].min()
        if pd.notna(fw): forecast_warning = int(fw)
        if pd.notna(ff): forecast_failure = int(ff)

    ticket = generate_ticket(module, signal, trend_df.iloc[-1], forecast_warning, forecast_failure)
    return trend_df, forecast_df, ticket, warning_threshold, failure_threshold

# =========================
# PAGES
# =========================
df = load_ai4i()

if selected_page == "Overview":
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(df))
    c2.metric("Machine Failures", int(df["Machine failure"].sum()))
    c3.metric("Failure Rate", f"{df['Machine failure'].mean():.2%}")
    c4.metric("Product Types", df["Type"].nunique())

    st.markdown("""
    ### Core AI Modules
    - **Machine Health Monitoring:** observe speed, torque, temperature, and tool wear.
    - **Failure Prediction:** classify whether a machine is likely to fail.
    - **Failure Trend Prediction:** detect early warning, watch, and failure zones.
    - **Autonomous Maintenance Ticketing:** assign priority, owner, due time, and action.
    - **Telegram / Email Alerts:** notify maintenance teams from the dashboard.
    """)
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Failure Type Distribution")
    failure_counts = df["Failure Type"].value_counts().reset_index()
    failure_counts.columns = ["Failure Type", "Count"]
    st.plotly_chart(px.bar(failure_counts, x="Failure Type", y="Count", title="Failure Type Distribution"), use_container_width=True)

elif selected_page == "Machine Health":
    st.subheader("Machine Health Monitoring")
    product_filter = st.sidebar.multiselect("Product Type", sorted(df["Type"].unique()), default=sorted(df["Type"].unique()))
    view = df[df["Type"].isin(product_filter)]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.scatter(view, x="Rotational speed [rpm]", y="Torque [Nm]", color="Machine failure", title="Speed vs Torque"), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(view, x="Process temperature [K]", y="Tool wear [min]", color="Machine failure", title="Temperature vs Tool Wear"), use_container_width=True)

    st.plotly_chart(px.line(view, x="Sample_Index", y=["Air temperature [K]", "Process temperature [K]"], title="Temperature Trend"), use_container_width=True)
    st.plotly_chart(px.line(view, x="Sample_Index", y=["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"], title="Machine Sensor Trends"), use_container_width=True)

elif selected_page == "Failure Prediction Demo":
    st.subheader("Live Machine Failure Prediction Demo")

    model_path = MODELS / "machine_failure_model.joblib"

    from train_models import train_model

    try:
        if not model_path.exists():
            st.warning("Training AI model for first deployment...")
            train_model()

        model = joblib.load(model_path)

    except Exception as e:
        st.warning("Existing model is incompatible. Retraining model now...")
        train_model()
        model = joblib.load(model_path)

    metrics = load_metrics()
    render_model_evaluation(model, metrics)

    st.subheader("Try a Live Prediction")

    sample = df.drop(
        columns=[
            "UDI",
            "Product ID",
            "Machine failure",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF",
            "Failure Type",
        ],
        errors="ignore",
    ).iloc[[0]].copy()

    edited = st.data_editor(sample, num_rows="fixed")

    if st.button("Predict Machine Failure"):
        pred = int(model.predict(edited)[0])
        proba = (
            model.predict_proba(edited)[0][1]
            if hasattr(model, "predict_proba")
            else pred
        )

        risk_level = classify_maintenance_risk(float(proba))

        col1, col2, col3 = st.columns(3)
        col1.metric("Failure Probability", f"{proba:.2%}")
        col2.metric("Maintenance Risk Level", risk_level)
        col3.metric("Predicted Class", "Failure" if pred == 1 else "Normal")

        if pred == 1:
            st.error(f"Predicted Result: MACHINE FAILURE RISK ({proba:.2%})")
            st.warning("Recommended Action: Inspect this machine within 24-48 hours and prioritize preventive maintenance.")
        else:
            st.success(f"Predicted Result: NORMAL ({proba:.2%} failure probability)")
            st.info("Recommended Action: Continue routine monitoring and scheduled preventive maintenance.")
            
elif selected_page == "Failure Trend Prediction":
    st.subheader("Failure Trend Prediction Before Breakdown")
    signal_map = get_signal_map()
    module = st.selectbox("Choose machine module", list(signal_map.keys()))
    signal_options = signal_map[module]
    signal_names = [x[0] for x in signal_options]
    signal = st.selectbox("Choose signal", signal_names)
    direction = dict(signal_options)[signal]
    direction = st.selectbox("Failure direction", [direction, "above" if direction == "below" else "below"], index=0)
    window = st.slider("Rolling window", 3, 50, 10)
    periods = st.slider("Forecast future samples", 5, 100, 15)

    trend_df, forecast_df, ticket, warning_threshold, failure_threshold = get_latest_trend_ticket(module, signal, direction, window, periods)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Watch Points", int((trend_df["failure_status"] == "WATCH").sum()))
    c2.metric("Early Warnings", int((trend_df["failure_status"] == "EARLY WARNING").sum()))
    c3.metric("Failures", int((trend_df["failure_status"] == "FAILURE").sum()))
    c4.metric("Latest Severity", trend_df.iloc[-1]["severity_level"])

    fig = px.line(trend_df, x="Sample_Index", y=[signal, "rolling_mean"], title=f"{signal} Failure Trend")
    fig.add_hline(y=warning_threshold, line_dash="dash", annotation_text="Warning Threshold")
    fig.add_hline(y=failure_threshold, line_dash="dot", annotation_text="Failure Threshold")
    warning_points = trend_df[trend_df["failure_status"].isin(["EARLY WARNING", "FAILURE"])]
    if not warning_points.empty:
        points = px.scatter(warning_points, x="Sample_Index", y=signal, color="failure_status")
        for trace in points.data:
            fig.add_trace(trace)
    st.plotly_chart(fig, use_container_width=True)

    if not forecast_df.empty:
        fcol = f"forecast_{signal}"
        forecast_fig = px.line(forecast_df, x="Sample_Index", y=fcol, title="Future Forecast Trend")
        forecast_fig.add_hline(y=warning_threshold, line_dash="dash", annotation_text="Warning Threshold")
        forecast_fig.add_hline(y=failure_threshold, line_dash="dot", annotation_text="Failure Threshold")
        st.plotly_chart(forecast_fig, use_container_width=True)

    show_ticket(ticket)

elif selected_page == "AI Chatbot":
    st.subheader("AI Chatbot for Steel Manufacturing Maintenance")
    st.write("Ask about risky machines, failure trend, tool wear, torque, temperature, or maintenance action.")
    query = st.chat_input("Ask: Which signal is risky? What maintenance action is needed?")
    if query:
        q = query.lower()
        if "tool" in q:
            module, signal, direction = "Tool Wear System", "Tool wear [min]", "above"
        elif "torque" in q or "power" in q:
            module, signal, direction = "Motor / Drive System", "Torque [Nm]", "above"
        elif "temperature" in q or "heat" in q:
            module, signal, direction = "Thermal System", "Process temperature [K]", "above"
        elif "speed" in q or "rpm" in q:
            module, signal, direction = "Motor / Drive System", "Rotational speed [rpm]", "below"
        else:
            module, signal, direction = "Overall Machine", "Mechanical power proxy", "above"

        trend_df, forecast_df, ticket, warning_threshold, failure_threshold = get_latest_trend_ticket(module, signal, direction)
        response = f"""
### AI Chatbot Analysis

**Module:** {module}  
**Signal analysed:** {signal}  
**Current status:** {ticket['failure_status']}  
**Priority:** {ticket['priority']}  
**Risk probability:** {ticket['risk_probability']}  
**Severity score:** {ticket['severity_score']}  

**Likely cause:** {ticket['likely_cause']}  

**Recommended action:**  
{ticket['maintenance_action']}
"""
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.dataframe(trend_df.tail(20), use_container_width=True)

elif selected_page == "Maintenance Tickets":
    st.subheader("Maintenance Ticket Backlog")
    rows = []
    for module, signals in get_signal_map().items():
        for signal, direction in signals:
            trend_df, forecast_df, ticket, _, _ = get_latest_trend_ticket(module, signal, direction)
            rows.append(ticket)
    tickets = pd.DataFrame(rows).sort_values(["priority", "severity_score"], ascending=[True, False])
    st.dataframe(tickets, use_container_width=True)
    st.download_button("Download Ticket Backlog CSV", tickets.to_csv(index=False), "steel_maintenance_ticket_backlog.csv", "text/csv")
