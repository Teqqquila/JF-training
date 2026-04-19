#!/usr/bin/env python3
"""
Jellyfin Recommender Training Manager API
==========================================
FastAPI backend that:
- Triggers retraining via retrain.py
- Pulls metrics from MLflow
- Manages scheduled retraining
- Serves recommendations via inference engine
- Handles user/admin login
"""

import os
import sys
import json
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/app")

from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from mlflow.tracking import MlflowClient

app = FastAPI(title="Jellyfin Recommender Training Manager")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ──
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.107:30500")
EXPERIMENT_NAME = "indieflicks-recommender"
RETRAIN_SCRIPT = "scripts/retrain.py"
CONFIG_PATH = "configs/config.yaml"
ADMIN_PASSWORD = "admin123"

mlflow.set_tracking_uri(MLFLOW_URI)

# ── State ──
training_state = {
    "status": "idle",
    "logs": [],
    "started_at": None,
    "finished_at": None,
}

schedule_state = {
    "enabled": False,
    "interval": "daily",
    "time": "02:00",
}


# ── Helpers ──

def get_mlflow_client():
    return MlflowClient(tracking_uri=MLFLOW_URI)


def get_experiment_runs():
    client = get_mlflow_client()
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return []
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
        return runs
    except Exception as e:
        print(f"MLflow error: {e}")
        return []


def run_to_dict(run):
    metrics = run.data.metrics
    params = run.data.params
    info = run.info
    return {
        "run_id": info.run_id,
        "run_name": info.run_name or params.get("model_type", "unknown"),
        "status": info.status,
        "start_time": datetime.fromtimestamp(info.start_time / 1000).strftime("%Y-%m-%d %H:%M") if info.start_time else None,
        "end_time": datetime.fromtimestamp(info.end_time / 1000).strftime("%Y-%m-%d %H:%M") if info.end_time else None,
        "data_version": params.get("data_version", "unknown"),
        "model_type": params.get("model_type", "unknown"),
        "epochs": params.get("epochs", "?"),
        "pretrained": params.get("pretrained", "false"),
        "metrics": {
            "best_val_mse": metrics.get("best_val_mse"),
            "final_val_mse": metrics.get("final_val_mse"),
            "hit_rate_10": metrics.get("hit_rate_10"),
            "ndcg_10": metrics.get("ndcg_10"),
            "total_wall_time_sec": metrics.get("total_wall_time_sec"),
        },
        "mlflow_url": f"{MLFLOW_URI}/#/experiments/{run.info.experiment_id}/runs/{info.run_id}",
    }


def run_retrain(version=None):
    global training_state
    training_state["status"] = "training"
    training_state["logs"] = ["Starting retraining..."]
    training_state["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_state["finished_at"] = None

    try:
        cmd = ["python", "-m", "scripts.retrain", "--config", CONFIG_PATH]
        if version:
            cmd += ["--version", version]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd="/app",
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                training_state["logs"].append(line)

        process.wait()

        if process.returncode == 0:
            training_state["status"] = "idle"
            training_state["logs"].append("Retraining completed successfully!")
        else:
            training_state["status"] = "error"
            training_state["logs"].append(f"Retraining failed with exit code {process.returncode}")

    except Exception as e:
        training_state["status"] = "error"
        training_state["logs"].append(f"Error: {str(e)}")

    training_state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Training API Routes ──

@app.get("/api/status")
def get_status():
    runs = get_experiment_runs()
    latest = run_to_dict(runs[0]) if runs else None
    return {
        "training_status": training_state["status"],
        "started_at": training_state["started_at"],
        "finished_at": training_state["finished_at"],
        "current_model": latest,
        "mlflow_uri": MLFLOW_URI,
    }


@app.get("/api/history")
def get_history():
    runs = get_experiment_runs()
    return {"runs": [run_to_dict(r) for r in runs]}


@app.get("/api/logs")
def get_logs():
    return {
        "status": training_state["status"],
        "logs": training_state["logs"],
    }


@app.post("/api/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    if training_state["status"] == "training":
        return JSONResponse(status_code=409, content={"error": "Training already in progress"})
    background_tasks.add_task(run_retrain)
    return {"message": "Retraining started", "status": "training"}


@app.post("/api/retrain/{version}")
def trigger_retrain_version(version: str, background_tasks: BackgroundTasks):
    if training_state["status"] == "training":
        return JSONResponse(status_code=409, content={"error": "Training already in progress"})
    background_tasks.add_task(run_retrain, version)
    return {"message": f"Retraining started with data {version}", "status": "training"}


@app.get("/api/schedule")
def get_schedule():
    return schedule_state


@app.post("/api/schedule")
def update_schedule(enabled: bool = None, interval: str = None, time_utc: str = None):
    if enabled is not None:
        schedule_state["enabled"] = enabled
    if interval is not None:
        schedule_state["interval"] = interval
    if time_utc is not None:
        schedule_state["time"] = time_utc
    return schedule_state


@app.post("/api/rollback/{version}")
def rollback_model(version: str):
    import boto3
    from botocore.client import Config as BotoConfig
    import yaml

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    s3 = config["s3"]
    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT", s3["endpoint"]),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", s3["access_key_id"]),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", s3["secret_access_key"]),
        region_name=s3.get("region", "us-east-1"),
        config=BotoConfig(signature_version="s3v4"),
    )

    src_key = f"models/mlp/{version}/model_mlp_best.pt"
    dst_key = "models/mlp/latest/model_mlp_best.pt"
    bucket = "warehouse"

    try:
        local_path = f"/tmp/rollback_{version}.pt"
        client.download_file(bucket, src_key, local_path)
        client.upload_file(local_path, bucket, dst_key)
        os.remove(local_path)
        training_state["logs"].append(f"Rolled back to {version}: copied {src_key} -> {dst_key}")
        return {"message": f"Rolled back to {version}", "status": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/datasets")
def list_datasets():
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    s3 = config["s3"]
    from scripts.data_loader import S3Config, build_boto3_s3_client, list_versions
    s3_cfg = S3Config(
        endpoint=os.environ.get("S3_ENDPOINT", s3["endpoint"]),
        access_key_id=os.environ.get("S3_ACCESS_KEY", s3["access_key_id"]),
        secret_access_key=os.environ.get("S3_SECRET_KEY", s3["secret_access_key"]),
        region=s3.get("region", "us-east-1"),
        use_ssl=s3.get("use_ssl", False),
    )
    client = build_boto3_s3_client(s3_cfg)
    root_dir = config["data"]["root_dir"]
    versions = list_versions(client, root_dir)
    current = config["data"].get("version", "auto")
    return {"versions": versions, "current": current}


# ── Inference Engine ──

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        import yaml
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        s3 = config["s3"]
        from api.inference import RecommendationEngine
        from scripts.data_loader import S3Config
        s3_cfg = S3Config(
            endpoint=os.environ.get("S3_ENDPOINT", s3["endpoint"]),
            access_key_id=os.environ.get("S3_ACCESS_KEY", s3["access_key_id"]),
            secret_access_key=os.environ.get("S3_SECRET_KEY", s3["secret_access_key"]),
            region=s3.get("region", "us-east-1"),
            use_ssl=s3.get("use_ssl", False),
        )
        _engine = RecommendationEngine(s3_cfg, config["data"])
        _engine.load_movies_csv()
        _engine.load_embeddings()
        _engine.load_model()
    return _engine


# ── Auth & Inference API Routes ──

@app.post("/api/login")
def login(role: str = "user", user_id: int = None, password: str = None):
    if role == "admin":
        if password != ADMIN_PASSWORD:
            return JSONResponse(status_code=401, content={"error": "Invalid admin password"})
        eng = get_engine()
        return {"role": "admin", "users": eng.get_user_ids(), "models": eng.get_model_versions()}
    elif role == "user":
        if user_id is None:
            return JSONResponse(status_code=400, content={"error": "user_id required"})
        eng = get_engine()
        if False:  # Any user ID is valid now
            return JSONResponse(status_code=404, content={"error": f"User {user_id} not found"})
        return {"role": "user", "user_id": user_id}
    return JSONResponse(status_code=400, content={"error": "Invalid role"})


@app.get("/api/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10, model_version: str = "latest"):
    eng = get_engine()
    model_key = f"models/mlp/{model_version}/model_mlp_best.pt"
    results, error = eng.recommend(user_id, top_n=top_n, model_key=model_key)
    if error:
        return JSONResponse(status_code=404, content={"error": error})
    return {"user_id": user_id, "model_version": model_version, "recommendations": results}


@app.get("/api/users")
def list_users():
    eng = get_engine()
    return {"users": eng.get_user_ids()}


@app.get("/api/model_versions")
def list_model_versions():
    eng = get_engine()
    return {"versions": eng.get_model_versions()}


# ── Serve frontend ──
frontend_dir = Path("/app/frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8096)