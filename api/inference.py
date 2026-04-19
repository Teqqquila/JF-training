"""
Inference engine - fetches candidates from API, reranks with MLP model.
"""
import os
import io
import csv
import numpy as np
import torch
import sys
import requests

sys.path.insert(0, "/app")

from scripts.data_loader import S3Config, build_boto3_s3_client, build_arrow_s3_filesystem
from scripts.retrain import RecommenderMLP


CANDIDATE_API = "http://10.43.252.8:18080/candidates"


class RecommendationEngine:
    def __init__(self, s3_cfg, data_cfg):
        self.s3_cfg = s3_cfg
        self.data_cfg = data_cfg
        self.client = build_boto3_s3_client(s3_cfg)

        self.model = None
        self.movie_info = {}
        self.loaded_model_key = None

    def load_movies_csv(self):
        print("Loading movies.csv...")
        obj = self.client.get_object(Bucket='raw', Key='movies.csv')
        content = obj['Body'].read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            mid = int(row['movieId'])
            self.movie_info[mid] = {
                'title': row['title'],
                'genres': row['genres'],
            }
        print(f"  Loaded {len(self.movie_info)} movies")

    def load_embeddings(self, version=None):
        print("Embeddings will be fetched from candidate API on demand")

    def load_model(self, model_key="models/mlp/latest/model_mlp_best.pt"):
        if self.loaded_model_key == model_key and self.model is not None:
            return

        print(f"Loading model from s3://warehouse/{model_key}...")
        local_path = "/tmp/inference_model.pt"
        self.client.download_file("warehouse", model_key, local_path)

        self.model = RecommenderMLP(
            embedding_dim=self.data_cfg.get('embedding_dim', 384),
            hidden_dims=[512, 256, 128],
            dropout=0.0,
        )

        state_dict = torch.load(local_path, map_location='cpu', weights_only=True)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = f"net.{k}" if not k.startswith("net.") else k
            new_state_dict[new_key] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.loaded_model_key = model_key
        print("  Model loaded!")

    def recommend(self, user_id, top_n=10, model_key=None):
        if model_key:
            self.load_model(model_key)
        elif self.model is None:
            self.load_model()

        # Fetch candidates from teammate's API
        try:
            resp = requests.get(
                CANDIDATE_API,
                params={"user_id": user_id, "top_k": 50},
                headers={"Accept": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return None, f"Candidate API error: {str(e)}"

        items = data.get("items", [])
        if not items:
            return None, f"No candidates for user {user_id}"

        category = data.get("category", "popular")
        has_embeddings = "embedding" in items[0] and "user_embedding" in data

        if has_embeddings:
            # MLP reranking: user has embedding, movies have embeddings
            user_emb = np.array(data["user_embedding"], dtype=np.float32)
            movie_ids = [item["movie_id"] for item in items]
            movie_embs = np.array([item["embedding"] for item in items], dtype=np.float32)
            user_embs = np.tile(user_emb, (len(movie_ids), 1))

            with torch.no_grad():
                user_t = torch.tensor(user_embs, dtype=torch.float32)
                movie_t = torch.tensor(movie_embs, dtype=torch.float32)
                scores = self.model(user_t, movie_t).numpy()

            ranked_idx = np.argsort(scores)[::-1][:top_n]

            results = []
            for idx in ranked_idx:
                mid = movie_ids[idx]
                info = self.movie_info.get(mid, {})
                results.append({
                    'movie_id': mid,
                    'title': info.get('title', f'Movie {mid}'),
                    'genres': info.get('genres', 'Unknown'),
                    'score': float(scores[idx]),
                    'method': 'mlp_reranking',
                    'candidate_rank': items[idx].get('rank', 0),
                    'candidate_score': float(items[idx].get('score', 0)),
                })
        else:
            # No embeddings - use candidate API scores directly (popular/fallback)
            sorted_items = sorted(items, key=lambda x: x.get('score', 0), reverse=True)[:20]

            results = []
            for i, item in enumerate(sorted_items):
                mid = item["movie_id"]
                info = self.movie_info.get(mid, {})
                results.append({
                    'movie_id': mid,
                    'title': info.get('title', f'Movie {mid}'),
                    'genres': info.get('genres', 'Unknown'),
                    'score': float(item.get('score', 0)),
                    'method': 'popular_fallback',
                    'candidate_rank': item.get('rank', i + 1),
                    'candidate_score': float(item.get('score', 0)),
                })

        return results, None

    def get_user_ids(self):
        return [37257905,32218290,40262109,13637575,82016510,72043515,93240786,89131271,94374605,13471527,12404661,29348212,41348220,63798927,77349752,932407]

    def get_model_versions(self):
        resp = self.client.list_objects_v2(
            Bucket='warehouse', Prefix='models/mlp/', Delimiter='/'
        )
        versions = []
        for p in resp.get('CommonPrefixes', []):
            name = p['Prefix'].split('/')[-2]
            if name != 'latest':
                versions.append(name)
        versions.append('latest')
        return versions