#!/usr/bin/env python3
"""
IndieFlicks Recommender Retraining Script
==========================================
Fine-tunes MLP on new user data streamed from MinIO (S3).
Loads previous best weights, trains on new version, saves versioned + latest model.

Usage:
    python -m scripts.train --config configs/config.yaml
"""

import argparse
import os
import time
from datetime import datetime
import platform

import boto3
from botocore.client import Config
import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml

from scripts.data_loader import (
    S3Config,
    VersionedDatasetConfig,
    build_versioned_dataloader,
)


# ──────────────────────────────────────────────
# MLP Model
# ──────────────────────────────────────────────

class RecommenderMLP(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        input_dim = embedding_dim * 2 + 3

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, user_emb, movie_emb):
        cosine_sim = torch.sum(user_emb * movie_emb, dim=1, keepdim=True) / (
            torch.norm(user_emb, dim=1, keepdim=True) *
            torch.norm(movie_emb, dim=1, keepdim=True) + 1e-8
        )
        dot_product = torch.sum(user_emb * movie_emb, dim=1, keepdim=True)
        l2_dist = torch.norm(user_emb - movie_emb, dim=1, keepdim=True)

        x = torch.cat([user_emb, movie_emb, cosine_sim, dot_product, l2_dist], dim=1)
        return self.net(x).squeeze(1)


# ──────────────────────────────────────────────
# S3 Model Persistence
# ──────────────────────────────────────────────

def build_s3_client(s3_cfg: S3Config):
    return boto3.client(
        "s3",
        endpoint_url=s3_cfg.endpoint,
        aws_access_key_id=s3_cfg.access_key_id,
        aws_secret_access_key=s3_cfg.secret_access_key,
        region_name=s3_cfg.region,
        config=Config(signature_version="s3v4"),
    )


def download_weights_from_s3(s3_cfg, bucket, key, local_path):
    """Download pretrained weights from MinIO."""
    print(f"  Downloading weights from s3://{bucket}/{key} ...")
    client = build_s3_client(s3_cfg)
    client.download_file(bucket, key, local_path)
    print(f"  Saved to {local_path}")


def upload_model_to_s3(s3_cfg, local_path, bucket, key):
    """Upload trained model to MinIO."""
    print(f"  Uploading model to s3://{bucket}/{key} ...")
    client = build_s3_client(s3_cfg)
    client.upload_file(local_path, bucket, key)
    print(f"  Done.")


# ──────────────────────────────────────────────
# Data Loader Builder
# ──────────────────────────────────────────────

def build_loader(s3_cfg, data_cfg, split, shuffle=False, return_ids=False):
    dataset_cfg = VersionedDatasetConfig(
        root_dir=data_cfg["root_dir"],
        split=split,
        version=data_cfg.get("version"),
        parquet_batch_size_rows=data_cfg.get("parquet_batch_size_rows", 4096),
        return_reference_fields=return_ids,
        shuffle_files=shuffle,
    )

    loader = build_versioned_dataloader(
        s3_cfg=s3_cfg,
        dataset_cfg=dataset_cfg,
        loader_batch_size=data_cfg.get("loader_batch_size", 256),
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=False,
    )
    print(f"  [{split}] version: {loader.dataset.version}, path: {loader.dataset.split_path}")
    return loader


# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    max_batches=None, log_interval=50):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        user_emb = batch["user_embedding"].to(device, non_blocking=True)
        movie_emb = batch["movie_embedding"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        pred = model(user_emb, movie_emb)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = label.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_samples
            print(f"    batch {batch_idx+1}/{max_batches or '?'}  "
                  f"samples={total_samples:,}  avg_mse={avg_loss:.6f}")

    if total_samples > 0:
        print(f"    Total: {total_samples:,} samples trained")

    return total_loss / max(total_samples, 1)


def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_labels = [], []
    all_user_ids, all_movie_ids = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break

            user_emb = batch["user_embedding"].to(device, non_blocking=True)
            movie_emb = batch["movie_embedding"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            pred = model(user_emb, movie_emb)
            loss = criterion(pred, label)

            bs = label.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            if "user_id" in batch:
                all_user_ids.append(batch["user_id"].numpy())
            if "movie_id" in batch:
                all_movie_ids.append(batch["movie_id"].numpy())

    result = {
        "val_mse": total_loss / max(total_samples, 1),
        "val_samples": total_samples,
        "preds": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
    }
    if all_user_ids:
        result["user_ids"] = np.concatenate(all_user_ids)
    if all_movie_ids:
        result["movie_ids"] = np.concatenate(all_movie_ids)
    return result


# ──────────────────────────────────────────────
# Ranking Metrics
# ──────────────────────────────────────────────

def compute_ranking_metrics(user_ids, movie_ids, labels, scores, k=10):
    import pandas as pd

    df = pd.DataFrame({
        "user_id": user_ids, "movie_id": movie_ids,
        "label": labels, "pred_score": scores,
    })

    results = []
    for _, group in df.groupby("user_id"):
        if len(group) < 2:
            continue
        top_k = group.sort_values("pred_score", ascending=False).head(k)
        true_top = set(group.sort_values("label", ascending=False).head(k)["movie_id"].values)

        hit_rate = 1.0 if len(set(top_k["movie_id"].values) & true_top) > 0 else 0.0

        relevances = top_k["label"].values
        ideal = np.sort(group["label"].values)[::-1][:k]
        dcg = np.sum(relevances / np.log2(np.arange(2, len(relevances) + 2)))
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        results.append({"hit_rate": hit_rate, "ndcg": ndcg})

    if not results:
        return {"hit_rate_10": 0.0, "ndcg_10": 0.0}

    avg = pd.DataFrame(results).mean()
    return {"hit_rate_10": float(avg["hit_rate"]), "ndcg_10": float(avg["ndcg"])}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IndieFlicks MLP Retraining")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--version", type=str, default=None, help="Override data version")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # S3
    s3_cfg = S3Config(
        endpoint=os.environ.get("S3_ENDPOINT", config["s3"]["endpoint"]),
        access_key_id=os.environ.get("S3_ACCESS_KEY", config["s3"]["access_key_id"]),
        secret_access_key=os.environ.get("S3_SECRET_KEY", config["s3"]["secret_access_key"]),
        region=config["s3"].get("region", "us-east-1"),
        use_ssl=config["s3"].get("use_ssl", False),
    )

    # MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    print(f"MLflow tracking: {tracking_uri}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data version
    data_cfg = config["data"]
    data_version = args.version or data_cfg.get("version", "auto")
    if args.version:
        data_cfg["version"] = args.version
    print(f"Data version: {data_version}")

    # Data loaders
    print("\n--- Building data loaders ---")
    train_loader = build_loader(s3_cfg, data_cfg, "train", shuffle=True)
    val_loader = build_loader(s3_cfg, data_cfg, "val", shuffle=False, return_ids=True)

    # Model
    model_cfg = config["models"]["mlp"]["params"]
    model = RecommenderMLP(
        embedding_dim=data_cfg.get("embedding_dim", 384),
        hidden_dims=model_cfg["hidden_dims"],
        dropout=model_cfg["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Load pretrained weights
    pretrained_cfg = config.get("pretrained")
    if pretrained_cfg and pretrained_cfg.get("enabled", False):
        local_weight_path = "./pretrained_weights.pt"

        source = pretrained_cfg.get("source", "local")
        if source == "s3":
            download_weights_from_s3(
                s3_cfg,
                pretrained_cfg["s3_bucket"],
                pretrained_cfg["s3_key"],
                local_weight_path,
            )
        elif source == "local":
            local_weight_path = pretrained_cfg["local_path"]

        print(f"  Loading pretrained weights from {local_weight_path}")
        state_dict = torch.load(local_weight_path, map_location=device, weights_only=True)

        # Remap keys: old weights use '0.weight', new model expects 'net.0.weight'
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = f"net.{k}" if not k.startswith("net.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        print(f"  Pretrained weights loaded!")
    else:
        print("  Training from scratch (no pretrained weights)")

    # Training config
    max_train_batches = config.get("training", {}).get("max_train_batches")
    max_val_batches = config.get("training", {}).get("max_val_batches")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_cfg["learning_rate"],
        weight_decay=model_cfg["weight_decay"],
    )
    epochs = model_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    print(f"\n--- Retraining on {data_version} ---")
    print(f"  Epochs: {epochs}")
    print(f"  Max train batches/epoch: {max_train_batches or 'all'}")
    print(f"  Max val batches: {max_val_batches or 'all'}")

    # MLflow run
    run_name = datetime.now().strftime('v%Y%m%d')
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "mlp")
        mlflow.log_param("device", str(device))
        mlflow.log_param("platform", platform.node())
        mlflow.log_param("param_count", param_count)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("pretrained", pretrained_cfg.get("enabled", False) if pretrained_cfg else False)
        mlflow.log_param("pretrained_source", pretrained_cfg.get("s3_key", "none") if pretrained_cfg else "none")
        mlflow.log_param("max_train_batches", max_train_batches or "all")
        mlflow.log_param("max_val_batches", max_val_batches or "all")
        for k, v in model_cfg.items():
            mlflow.log_param(k, v)

        best_val_mse = float("inf")
        model_save_path = "./model_mlp_best.pt"
        total_start = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            train_mse = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                max_batches=max_train_batches,
            )

            val_result = evaluate(
                model, val_loader, criterion, device,
                max_batches=max_val_batches,
            )
            val_mse = val_result["val_mse"]

            scheduler.step()
            epoch_time = time.time() - epoch_start

            print(f"  Epoch {epoch:3d}/{epochs}  train_mse={train_mse:.6f}  "
                  f"val_mse={val_mse:.6f}  time={epoch_time:.1f}s")

            mlflow.log_metrics({
                "train_mse": train_mse,
                "val_mse": val_mse,
                "epoch_time_sec": epoch_time,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save(model.state_dict(), model_save_path)
                print(f"    -> New best (val_mse={val_mse:.6f})")

        total_time = time.time() - total_start

        # Final eval with best model
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        final_val = evaluate(model, val_loader, criterion, device, max_batches=max_val_batches)

        final_metrics = {
            "best_val_mse": best_val_mse,
            "final_val_mse": final_val["val_mse"],
            "total_wall_time_sec": total_time,
            "epochs_trained": epochs,
        }

        if "user_ids" in final_val and "movie_ids" in final_val:
            ranking = compute_ranking_metrics(
                final_val["user_ids"], final_val["movie_ids"],
                final_val["labels"], final_val["preds"],
            )
            final_metrics.update(ranking)

        mlflow.log_metrics(final_metrics)
        mlflow.log_artifact(args.config)
        if os.path.exists(model_save_path):
            mlflow.log_artifact(model_save_path)

        # Upload model to S3 — versioned + latest
        model_persist = config.get("model_output")
        if model_persist:
            bucket = model_persist["s3_bucket"]

            # Save to version directory (e.g. models/mlp/v0002/)
            if "version_key" in model_persist:
                upload_model_to_s3(s3_cfg, model_save_path, bucket, model_persist["version_key"])

            # Update latest
            if "latest_key" in model_persist:
                upload_model_to_s3(s3_cfg, model_save_path, bucket, model_persist["latest_key"])

        print(f"\n{'='*50}")
        print(f"Retraining complete! (data: {data_version})")
        print(f"Total time: {total_time:.1f}s")
        for k, v in sorted(final_metrics.items()):
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
