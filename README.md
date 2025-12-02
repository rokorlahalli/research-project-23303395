📌 Machine Learning based Ensemble Anomaly Detection in Kuberentes Environments

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

⚙️ Tech Stack

Python, FastAPI, Uvicorn

Prometheus, Pushgateway, Grafana

Google Kubernetes Engine (GKE)

SageMaker (for model training)

Docker

Ollama / LLMs (optional)

🚀 Deployment Guide
1️⃣ Build & Push Docker Image
```bash
docker build -t docker pull rokorlahalli/ric-ensemble:v1 .
docker push docker pull rokorlahalli/ric-ensemble:v1
```
2️⃣ Apply Kubernetes Manifests
```bash
kubectl apply -f manifests/pushgateway.yaml
kubectl apply -f manifests/deployment.yaml
kubectl apply -f manifests/service.yaml
kubectl apply -f manifests/grafana/
kubectl apply -f manifests/prometheus/
```
3️⃣ Verify Pods
```bash
kubectl get pods -n <your-app-namespace>
```
4️⃣ Test API
```bash
curl "<api-url>/predict?cpu=0.5&memory=0.7&..."
```
👨‍💻 Author
Rohit Korlahalli
Student ID - 23303395
MSc Cloud Computing — National College of Ireland (NCI)
