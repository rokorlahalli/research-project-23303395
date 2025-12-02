from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import requests
import os
import csv
import re
import pandas as pd
import httpx
from datetime import datetime, timezone, timedelta
from fastapi_utils.tasks import repeat_every
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware

last_low_alert_time = None
# Memory for deduplication
recent_insights = defaultdict(lambda: {"text": None, "last_sent": None})
INSIGHT_COOLDOWN_MINUTES = 30  # do not resend same insight before 30 mins

# APP INITIALIZATION
app = FastAPI(title="Kubernetes Ensemble Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD MODELS
print("Loading models...")
iforest = joblib.load("iforest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
lstm_model = tf.keras.models.load_model("lstm_model.h5")

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# ENVIRONMENT VARIABLES
PUSHGATEWAY_URL = os.getenv(
    "PUSHGATEWAY_URL",
    "http://pushgateway-prometheus-pushgateway.prometheus.svc.cluster.local:9091"
)
PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://kps-kube-prometheus-stack-prometheus.prometheus.svc.cluster.local:9090"
)
OLLAMA_URL = "http://devops-assistant.app.svc.cluster.local:11434/api/generate"
OLLAMA_TIMEOUT = 180.0
CSV_LOG_PATH = os.getenv("CSV_LOG_PATH", "/data/predictions_log.csv")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# INPUT SCHEMA
class TelemetryInput(BaseModel):
    cpu_usage: float
    memory_usage: float
    net_receive: float
    net_transmit: float
    fs_reads_bytes: float
    fs_writes_bytes: float
    restarts: float
    pod_name: str = "unknown"

#Devops Assistant Function

@app.get("/ask-ai")
async def ask_ai(query: str):
    try:
        # Create httpx client with 180 second timeout
        timeout = httpx.Timeout(180.0, connect=10.0)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": "llama3.2:1b",
                    "prompt": query,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                return JSONResponse(
                    status_code=500,
                    content={"data": [{"answer": f"Ollama error: {response.status_code}"}]}
                )
            
            ollama_data = response.json()
            answer = ollama_data.get("response", "No response")
            
            return {"data": [{"answer": answer}]}
            
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"data": [{"answer": "Request timed out. Please try a simpler question."}]}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"data": [{"answer": f"Error: {str(e)}"}]}
        )

@app.post("/ask-ai")
def ask_ai_post(request: dict):
    query = request.get("query", "")
    if not query:
        return {"error": "Query text is required"}

    payload = {
        "model": "llama3.2:1b",
        "prompt": query,
        "stream": False     # FIXED HERE ALSO
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        data = response.json()
        return {"answer": data.get("response", "").strip()}

    except Exception as e:
        return {"error": str(e)}

# INSIGHT & PUSH HELPERS
def generate_insight(cpu, mem, prediction):
    """
    Returns human-readable insight + remediation suggestions.
    Adaptive thresholds ensure realistic anomaly behavior even under modest loads.
    Includes basic noise suppression for low-utilization insights.
    """
    global last_low_alert_time

    try:
        cpu = float(cpu) if not callable(cpu) else 0.0
        mem = float(mem) if not callable(mem) else 0.0
    except Exception:
        cpu, mem = 0.0, 0.0

    insights = []

    CPU_HIGH = 0.06     
    CPU_LOW = 0.01     
    MEM_HIGH = 0.05   
    MEM_LOW = 0.01

    now = datetime.utcnow()

    # --- CPU logic ---
    if cpu > CPU_HIGH:
        insights.append("High CPU utilization detected — consider enabling or adjusting HPA thresholds.")
    elif cpu < CPU_LOW:
        # suppress repeated low CPU alerts (only once/hour)
        if not last_low_alert_time or (now - last_low_alert_time).seconds > 3600:
            insights.append("Low CPU utilization — consider reducing resource requests for efficiency.")
            last_low_alert_time = now

    # --- Memory logic ---
    if mem > MEM_HIGH:
        insights.append("Memory nearing saturation — consider using VPA or increasing memory limits.")
    elif mem < MEM_LOW:
        # suppress repeated low memory alerts (only once/hour)
        if not last_low_alert_time or (now - last_low_alert_time).seconds > 3600:
            insights.append("Memory underutilized — review memory requests or optimize workloads.")
            last_low_alert_time = now

    # --- Prediction-based insight ---
    if prediction == 1:
        insights.append("Anomaly detected — investigate affected pod metrics and logs.")
    else:
        insights.append("System stable — no anomalies detected.")

    return " | ".join(insights)


def sanitize_label_value(value: str) -> str:
    """
    Cleans a string to make it Prometheus label-safe.
    Removes unsupported characters and limits length.
    """
    # Escape or strip unsafe characters
    # Place '-' at the END of the class so it’s not read as a range
    clean = re.sub(r'[^a-zA-Z0-9_:., \-]', '', value)
    # Replace spaces with underscores
    clean = clean.replace(" ", "_")
    # Collapse repeated underscores
    clean = re.sub(r'_+', '_', clean)
    # Truncate long strings (Prometheus label limit is 256 bytes)
    return clean[:250]


def push_insight_to_prometheus(pod_name, insight_text):
    """
    Pushes AI-generated insights to Prometheus Pushgateway,
    grouping metrics per pod instance to avoid overwriting.
    """
    try:
        metric_name = "anomaly_insight"
        safe_text = sanitize_label_value(insight_text)

        data = (
            f"# HELP {metric_name} AI-driven insight metric\n"
            f"# TYPE {metric_name} gauge\n"
            f"{metric_name}{{pod=\"{pod_name}\", insight=\"{safe_text}\"}} 1\n"
        )

        # Save the response from the PUT request
        response = requests.put(
            f"{PUSHGATEWAY_URL}/metrics/job/ai_insight/instance/{pod_name}",
            data=data,
            timeout=5
        )

        if response.status_code in (200, 202):
            print(f"Insight pushed for {pod_name}")
        else:
            print(f"⚠️ Push failed for {pod_name}: {response.status_code} | {response.text}")

    except Exception as e:
        print(f"⚠️ Error pushing insight to Prometheus: {e}")


def push_prediction_to_prometheus(pod_name, prediction):
    """
    Push numeric anomaly prediction metric to Prometheus
    (used for dashboard panels like anomaly trend and total anomalies).
    """
    try:
        metric_name = "anomaly_prediction"
        data = (
            f"# HELP {metric_name} Anomaly prediction result (1=Anomaly, 0=Normal)\n"
            f"# TYPE {metric_name} gauge\n"
            f"{metric_name}{{pod=\"{pod_name}\"}} {int(prediction)}\n"
        )

        response = requests.put(
            f"{PUSHGATEWAY_URL}/metrics/job/anomaly_prediction/instance/{pod_name}",
            data=data,
            timeout=5
        )

        if response.status_code in (200, 202):
            print(f"✅ Prediction metric pushed for {pod_name}: {prediction}")
        else:
            print(f"⚠️ Failed to push prediction metric: {response.status_code} | {response.text}")
    except Exception as e:
        print(f"⚠️ Error pushing prediction metric: {e}")


def send_slack_alert(pod_name, insight_text):
    if not SLACK_WEBHOOK_URL:
        return
    payload = {
        "text": f"🚨 *AIOps Insight*\n*Pod:* `{pod_name}`\n*Insight:* {insight_text}\n*Time:* {datetime.now(timezone.utc).isoformat()}"
    }
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        if resp.status_code not in (200, 204):
            print(f"⚠️ Slack webhook failed: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Slack alert error: {e}")

# PROMETHEUS QUERY HELPERS
def query_prometheus(expr: str, minutes: int = 2):
    end = datetime.utcnow()
    start = end - timedelta(minutes=minutes)
    query_range_url = f"{PROMETHEUS_URL}/api/v1/query_range"
    params = {"query": expr, "start": start.timestamp(), "end": end.timestamp(), "step": "30s"}
    response = requests.get(query_range_url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json().get("data", {}).get("result", [])
    return data

def fetch_live_metrics(namespace="app"):
    metric_map = {
        "cpu_usage": f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container!="",pod!=""}}[2m])',
        "memory_usage": f'container_memory_usage_bytes{{namespace="{namespace}",container!="",pod!=""}}',
        "net_receive": f'rate(container_network_receive_bytes_total{{namespace="{namespace}"}}[2m])',
        "net_transmit": f'rate(container_network_transmit_bytes_total{{namespace="{namespace}"}}[2m])',
        "fs_reads_bytes": f'rate(container_fs_reads_bytes_total{{namespace="{namespace}"}}[2m])',
        "fs_writes_bytes": f'rate(container_fs_writes_bytes_total{{namespace="{namespace}"}}[2m])',
        "restarts": f'rate(kube_pod_container_status_restarts_total{{namespace="{namespace}"}}[5m])'
    }

    pod_data = {}

    for metric, expr in metric_map.items():
        try:
            results = query_prometheus(expr)
            if not isinstance(results, list):
                print(f"⚠️ Unexpected response type for {metric}: {type(results)}")
                continue

            for r in results:
                pod = r.get("metric", {}).get("pod", "unknown")
                raw_values = r.get("values")

                # Defensive check for invalid structures
                if not raw_values or not isinstance(raw_values, list):
                    print(f"⚠️ No valid values for {metric} in pod {pod}")
                    continue

                # Prometheus usually returns a list of [timestamp, value]
                try:
                    last_timestamp, last_value = raw_values[-1]
                    last_value = float(last_value)
                except Exception as e:
                    print(f"⚠️ Conversion error for {metric} pod={pod}: {e}")
                    last_value = 0.0

                if pod not in pod_data:
                    pod_data[pod] = {}

                pod_data[pod][metric] = last_value

        except Exception as e:
            print(f"⚠️ Error fetching {metric}: {e}")

    if not pod_data:
        print("⚠️ No metrics collected — returning empty DataFrame.")
        return pd.DataFrame(columns=["pod_name"] + list(metric_map.keys()))

    # Debug print for inspection
    print("🔍 Sample Prometheus data (1st pod):", list(pod_data.items())[:1])

    df = pd.DataFrame.from_dict(pod_data, orient="index").fillna(0).reset_index()
    df.rename(columns={"index": "pod_name"}, inplace=True)
    return df


# CORE PREDICTION FUNCTION 
def run_prediction(df):
    expected_cols = [
        "cpu_usage", "memory_usage", "net_receive", "net_transmit",
        "fs_reads_bytes", "fs_writes_bytes", "restarts"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"⚠️ Missing columns in dataframe: {missing}")
        return

    global recent_insights
    now = datetime.utcnow()

    # Define separate cooldowns
    NORMAL_COOLDOWN_HOURS = 12   # 2 alerts per day max for normal
    ANOMALY_COOLDOWN_MINUTES = 30

    for _, row in df.iterrows():
        pod = row.get("pod_name", "unknown")

        try:
            # --- Safe numeric extraction ---
            def safe_float(value):
                try:
                    if callable(value):
                        return 0.0
                    return float(value)
                except Exception:
                    return 0.0

            cpu = safe_float(row.get("cpu_usage", 0.0))
            mem = safe_float(row.get("memory_usage", 0.0))
            net_recv = safe_float(row.get("net_receive", 0.0))
            net_tx = safe_float(row.get("net_transmit", 0.0))
            fs_read = safe_float(row.get("fs_reads_bytes", 0.0))
            fs_write = safe_float(row.get("fs_writes_bytes", 0.0))
            restarts = safe_float(row.get("restarts", 0.0))

            X = np.array([[cpu, mem, net_recv, net_tx, fs_read, fs_write, restarts]])

            # --- Scaling for each model ---
            X_std = standard_scaler.fit_transform(X)
            X_mm = minmax_scaler.fit_transform(X)
            X_lstm = np.expand_dims(np.vstack((np.zeros((29, X_mm.shape[1])), X_mm)), axis=0)

            # --- Ensemble predictions ---
            pred_if = np.where(iforest.predict(X_std) == -1, 1, 0)
            pred_xgb = xgb_model.predict(X_std)
            pred_lstm = (lstm_model.predict(X_lstm) > 0.5).astype(int).flatten()

            preds = np.array([pred_if[-1], pred_xgb[-1], pred_lstm[-1]])
            final_pred = int(np.sum(preds) >= 2)

            # --- Demo sensitivity: mild boost for visible anomalies ---
            if cpu > 0.06 or mem > 0.05:
                final_pred = 1

            # --- Generate insight text ---
            insight = generate_insight(cpu, mem, final_pred)

            # --- Retrieve last known info ---
            last_info = recent_insights.get(pod, {"text": None, "last_sent": None, "type": None})
            last_text = last_info["text"]
            last_time = last_info["last_sent"]
            last_type = last_info.get("type", None)

            is_new_insight = insight != last_text
            time_since_last = (now - last_time).total_seconds() / 60 if last_time else float("inf")

            # --- Determine cooldown dynamically ---
            if final_pred == 1:  # anomaly
                cooldown_expired = time_since_last > ANOMALY_COOLDOWN_MINUTES
                alert_type = "anomaly"
            else:
                cooldown_expired = time_since_last > NORMAL_COOLDOWN_HOURS * 60
                alert_type = "normal"

            # --- Always push metrics for Grafana ---
            push_insight_to_prometheus(pod, insight)

            push_prediction_to_prometheus(pod, final_pred)

            # --- Slack logic ---
            if is_new_insight or cooldown_expired:
                send_slack_alert(pod, insight)
                recent_insights[pod] = {"text": insight, "last_sent": now, "type": alert_type}
                print(f"📨 Slack alert sent for {pod} ({alert_type.upper()})")
            else:
                print(f"🤫 Skipped Slack alert for {pod} ({alert_type.upper()} cooldown active)")

            # --- Local log ---
            with open(CSV_LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([datetime.utcnow().isoformat(), pod, final_pred, insight])

            print(f"✅ {pod}: Prediction={final_pred}, Votes={preds}, Insight={insight}")

        except Exception as e:
            print(f"⚠️ Prediction failed for {pod}: {e}")

# AUTOMATED SCHEDULED JOB
@app.on_event("startup")
@repeat_every(seconds=300)  # every 5 minutes
def scheduled_prediction():
    try:
        df = fetch_live_metrics(namespace="app")
        if not df.empty:
            print(f"🔁 Running automated predictions for {len(df)} pods at {datetime.utcnow().isoformat()}")
            run_prediction(df)
        else:
            print("⚠️ No live pod metrics found.")
    except Exception as e:
        print(f"⚠️ Scheduled prediction error: {e}")

# HEALTH CHECK ENDPOINT
@app.get("/")
def health():
    return {"status": "ok", "message": "AI-enhanced Ensemble Anomaly Detection API is running with live metrics."}
