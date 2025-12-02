📌 Ensemble-Based Kubernetes Anomaly Detection Framework

A Lightweight ML + Observability Pipeline for Proactive AIOps

📖 Overview

This project implements a lightweight, ensemble-based anomaly detection system for Kubernetes clusters using Isolation Forest, XGBoost, and LSTM models. The framework integrates directly with Prometheus, Grafana, Pushgateway, and optionally an LLM-powered DevOps assistant (Ollama) to provide real-time anomaly detection, visualisation, and operational insights.

The system is deployed on Google Kubernetes Engine (GKE) and is designed to be scalable, interpretable, and easy to integrate with existing cloud-native observability stacks.

✨ Key Features

Ensemble ML-based anomaly detection (Isolation Forest + XGBoost + LSTM)

Real-time inference API built using FastAPI

Prometheus integration for ingesting pod/node metrics

Pushgateway used for publishing ML outputs as custom Prometheus metrics

Grafana dashboards for anomaly visualisation and cluster insights

Slack alerting for critical anomaly events

Optional LLM DevOps assistant for natural-language explanations and recommendations

Supports real + synthetic workloads for evaluation and testing

🧠 System Architecture
Prometheus → Ensemble API (FastAPI) → Pushgateway → Grafana
                              ↓
                      LLM DevOps Assistant (optional)
                              ↓
                            Slack Alerts

📂 Repository Structure
📁 ensemble-k8s-anomaly-detection
├── api/
│   ├── main.py                     # FastAPI inference service
│   ├── models/                     # Trained ML models (IF, XGB, LSTM)
│   └── utils/                      # Preprocessing, inference logic
│
├── manifests/
│   ├── deployment.yaml             # API deployment on GKE
│   ├── service.yaml                # ClusterIP / NodePort / Ingress
│   ├── pushgateway.yaml
│   ├── grafana/                    # Dashboards + datasources
│   └── prometheus/                 # Prometheus scrape configs
│
├── notebooks/
│   ├── training-iforest.ipynb
│   ├── training-xgboost.ipynb
│   ├── training-lstm.ipynb
│   └── ensemble-analysis.ipynb
│
├── frontend/ (optional)
│   └── devops-chat-ui/             # LLM assistant frontend
│
├── README.md
└── LICENSE (optional)

⚙️ Tech Stack

Python, FastAPI, Uvicorn

Prometheus, Pushgateway, Grafana

Google Kubernetes Engine (GKE)

SageMaker (for model training)

Docker

Ollama / LLMs (optional)

🚀 Deployment Guide
1️⃣ Build & Push Docker Image
docker build -t gcr.io/<project-id>/ensemble-api:v1 .
docker push gcr.io/<project-id>/ensemble-api:v1

2️⃣ Apply Kubernetes Manifests
kubectl apply -f manifests/pushgateway.yaml
kubectl apply -f manifests/deployment.yaml
kubectl apply -f manifests/service.yaml
kubectl apply -f manifests/grafana/
kubectl apply -f manifests/prometheus/

3️⃣ Verify Pods
kubectl get pods -n <namespace>

4️⃣ Test API
curl "<api-url>/predict?cpu=0.5&memory=0.7&..."

📊 Grafana Dashboards

Add your screenshots here:

📸 /screenshots/grafana-dashboard.png
📸 /screenshots/devops-assistant.png

📢 Slack Alerting (Optional)

Configure your Slack webhook:

SLACK_WEBHOOK_URL: "<your-webhook>"


Alerts are triggered when the ensemble predicts anomalies.

🧑‍💻 LLM DevOps Assistant (Optional)

The assistant provides:

Natural-language explanations

Recommendations (scale up/down, check node load, etc.)

Interactive Q&A based on cluster status

LLM endpoint example:

curl "http://<ingress-ip>/ask-ai?query=Why%20is%20my%20CPU%20spiking?"

📈 Model Evaluation

Performance metrics include:

Confusion matrices (IF, XGB, LSTM, Ensemble)

Macro F1-score

Anomaly detection latency

Stability under varying workloads

Refer to notebooks/ensemble-analysis.ipynb.

🔮 Future Enhancements

Add weighted ensemble voting

Introduce online learning for adapting to workload drift

Incorporate Kubernetes event logs into feature set

Expand LLM assistant into a full AIOps automation module

Build a plugin for Grafana-native anomaly queries

📄 License

MIT License (or whichever you choose)

👨‍💻 Author

Rohit Korlahalli
Student ID - 23303395
MSc Cloud Computing — National College of Ireland (NCI)
