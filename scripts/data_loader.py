import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Dict, Any

import boto3
from botocore.client import Config
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


# ============================================================
# Config
# ============================================================

@dataclass
class S3Config:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    region: str = "us-east-1"
    use_ssl: bool = False


@dataclass
class VersionedDatasetConfig:
    root_dir: str                      # e.g. s3://warehouse/dataset/versioned_dataset
    split: str = "train"               # train / val / test
    version: Optional[str] = None      # None -> auto latest valid version
    parquet_batch_size_rows: int = 4096
    return_reference_fields: bool = False
    shuffle_files: bool = False


# ============================================================
# Helpers
# ============================================================

_VERSION_RE = re.compile(r"^v(\d+)$")


def is_s3_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def split_s3_uri(uri: str):
    if not is_s3_path(uri):
        raise ValueError(f"Not an S3 URI: {uri}")
    no_scheme = uri[len("s3://"):]
    bucket, _, key = no_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def strip_s3_scheme(path: str) -> str:
    return path[len("s3://"):] if is_s3_path(path) else path


def s3_join(base: str, *parts: str) -> str:
    if not is_s3_path(base):
        raise ValueError(f"Expected S3 path, got: {base}")
    out = base.rstrip("/")
    for p in parts:
        out += "/" + p.strip("/")
    return out


def parse_version(v: str) -> int:
    m = _VERSION_RE.fullmatch(v)
    return int(m.group(1)) if m else -1


def build_boto3_s3_client(cfg: S3Config):
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name=cfg.region,
        config=Config(signature_version="s3v4"),
    )


def build_arrow_s3_filesystem(cfg: S3Config) -> pafs.S3FileSystem:
    scheme = "https" if cfg.use_ssl else "http"
    endpoint_no_scheme = cfg.endpoint.replace("http://", "").replace("https://", "")
    return pafs.S3FileSystem(
        access_key=cfg.access_key_id,
        secret_key=cfg.secret_access_key,
        endpoint_override=endpoint_no_scheme,
        scheme=scheme,
        region=cfg.region,
    )


def list_versions(s3_client, root_dir: str) -> List[str]:
    """
    按你测试脚本的逻辑列 versions/ 下的版本目录。
    """
    bucket, key = split_s3_uri(root_dir)
    prefix = key.rstrip("/") + "/versions/"

    resp = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        Delimiter="/",
    )

    versions = []
    for cp in resp.get("CommonPrefixes", []):
        name = cp["Prefix"].split("/")[-2]
        if _VERSION_RE.fullmatch(name):
            versions.append(name)

    return sorted(set(versions), key=parse_version)


def split_has_parquet_files(s3_client, root_dir: str, version: str, split: str) -> bool:
    """
    检查 <root_dir>/versions/<version>/<split>/ 下是否至少有一个 parquet 文件。
    """
    bucket, key = split_s3_uri(root_dir)
    prefix = f"{key.rstrip('/')}/versions/{version}/{split.strip('/')}/"

    resp = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=20,
    )

    for obj in resp.get("Contents", []):
        obj_key = obj["Key"]
        if obj_key.endswith(".parquet"):
            return True
    return False


def resolve_version(s3_client, root_dir: str, split: str, version: Optional[str]) -> str:
    """
    - 如果手动指定 version，则校验该 split 下是否有 parquet
    - 如果 version is None，则从大到小找最新且该 split 有 parquet 的版本
    """
    if version is not None:
        if parse_version(version) < 0:
            raise ValueError(f"Invalid version format: {version}")
        if not split_has_parquet_files(s3_client, root_dir, version, split):
            raise FileNotFoundError(
                f"No parquet files found under {root_dir}/versions/{version}/{split}"
            )
        return version

    versions = list_versions(s3_client, root_dir)
    if not versions:
        raise FileNotFoundError(f"No versions found under {root_dir}/versions/")

    for v in reversed(versions):
        if split_has_parquet_files(s3_client, root_dir, v, split):
            return v

    raise FileNotFoundError(
        f"No valid version contains parquet files for split={split} under {root_dir}/versions/"
    )


def build_split_path(root_dir: str, version: str, split: str) -> str:
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return s3_join(root_dir, "versions", version, split)


# ============================================================
# Dataset
# ============================================================

class VersionedParquetIterableDataset(IterableDataset):
    """
    读取:
      <root_dir>/versions/<latest_or_specified>/<split>/

    必需列:
      user_embedding, movie_embedding, label

    可选参考列:
      user_id, movie_id
    """
    def __init__(self, s3_cfg: S3Config, dataset_cfg: VersionedDatasetConfig):
        super().__init__()

        if not is_s3_path(dataset_cfg.root_dir):
            raise ValueError("This implementation currently expects root_dir to be an S3 URI")

        self.s3_cfg = s3_cfg
        self.dataset_cfg = dataset_cfg

        self.s3_client = build_boto3_s3_client(s3_cfg)
        self.fs = build_arrow_s3_filesystem(s3_cfg)

        self.version = resolve_version(
            s3_client=self.s3_client,
            root_dir=self.dataset_cfg.root_dir,
            split=self.dataset_cfg.split,
            version=self.dataset_cfg.version,
        )

        self.split_path = build_split_path(
            root_dir=self.dataset_cfg.root_dir,
            version=self.version,
            split=self.dataset_cfg.split,
        )

        self.required_columns = ["user_embedding", "movie_embedding", "label"]
        self.optional_columns = ["user_id", "movie_id"]

        dataset = self._build_dataset()
        schema_names = set(dataset.schema.names)

        missing = [c for c in self.required_columns if c not in schema_names]
        if missing:
            raise ValueError(
                f"Missing required columns in dataset {self.split_path}: {missing}. "
                f"Available columns: {sorted(schema_names)}"
            )

        self.available_optional_columns = [
            c for c in self.optional_columns if c in schema_names
        ]

    def _build_dataset(self) -> ds.Dataset:
        return ds.dataset(
            strip_s3_scheme(self.split_path),
            format="parquet",
            filesystem=self.fs,
        )

    def _columns_to_read(self) -> List[str]:
        cols = list(self.required_columns)
        if self.dataset_cfg.return_reference_fields:
            cols.extend(self.available_optional_columns)
        return cols

    def _get_worker_fragments(self, fragments: List[Any]) -> List[Any]:
        worker = get_worker_info()
        if worker is None:
            return fragments
        return fragments[worker.id::worker.num_workers]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        dataset = self._build_dataset()
        fragments = list(dataset.get_fragments())

        if self.dataset_cfg.shuffle_files and len(fragments) > 1:
            g = torch.Generator()
            g.manual_seed(torch.initial_seed())
            perm = torch.randperm(len(fragments), generator=g).tolist()
            fragments = [fragments[i] for i in perm]

        fragments = self._get_worker_fragments(fragments)
        columns = self._columns_to_read()

        for fragment in fragments:
            scanner = fragment.scanner(
                columns=columns,
                batch_size=self.dataset_cfg.parquet_batch_size_rows,
            )

            for record_batch in scanner.to_batches():
                pyd = record_batch.to_pydict()

                user_embedding = torch.tensor(pyd["user_embedding"], dtype=torch.float32)
                movie_embedding = torch.tensor(pyd["movie_embedding"], dtype=torch.float32)
                label = torch.tensor(pyd["label"], dtype=torch.float32)

                n = len(pyd["label"])
                for i in range(n):
                    sample = {
                        "user_embedding": user_embedding[i],
                        "movie_embedding": movie_embedding[i],
                        "label": label[i],
                    }

                    if self.dataset_cfg.return_reference_fields:
                        if "user_id" in pyd:
                            sample["user_id"] = torch.tensor(pyd["user_id"][i], dtype=torch.long)
                        if "movie_id" in pyd:
                            sample["movie_id"] = torch.tensor(pyd["movie_id"][i], dtype=torch.long)

                    yield sample


# ============================================================
# Collate
# ============================================================

def collate_training_batch(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Empty batch received")

    batch = {
        "user_embedding": torch.stack([x["user_embedding"] for x in samples], dim=0),
        "movie_embedding": torch.stack([x["movie_embedding"] for x in samples], dim=0),
        "label": torch.stack([x["label"] for x in samples], dim=0),
    }

    if "user_id" in samples[0]:
        batch["user_id"] = torch.stack([x["user_id"] for x in samples], dim=0)
    if "movie_id" in samples[0]:
        batch["movie_id"] = torch.stack([x["movie_id"] for x in samples], dim=0)

    return batch


# ============================================================
# Public Builder
# ============================================================

def build_versioned_dataloader(
    s3_cfg: S3Config,
    dataset_cfg: VersionedDatasetConfig,
    loader_batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = VersionedParquetIterableDataset(
        s3_cfg=s3_cfg,
        dataset_cfg=dataset_cfg,
    )

    return DataLoader(
        dataset,
        batch_size=loader_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_training_batch,
    )


# ============================================================
# Example test
# ============================================================

if __name__ == "__main__":
    s3_cfg = S3Config(
        endpoint="http://10.56.2.170:30900",
        access_key_id="minioadmin",
        secret_access_key="minioadmin123",
        region="eu-central-1",
        use_ssl=False,
    )

    dataset_cfg = VersionedDatasetConfig(
        root_dir="s3://warehouse/dataset/versioned_dataset",
        split="train",
        version="v0001",  # None -> 自动选最新且该 split 真有 parquet 的版本
        parquet_batch_size_rows=1024,
        return_reference_fields=False,
        shuffle_files=False,
    )

    loader = build_versioned_dataloader(
        s3_cfg=s3_cfg,
        dataset_cfg=dataset_cfg,
        loader_batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    print("resolved version:", loader.dataset.version)
    print("resolved split path:", loader.dataset.split_path)

    for batch in loader:
        print("user_embedding:", batch["user_embedding"].shape)
        print("movie_embedding:", batch["movie_embedding"].shape)
        print("label:", batch["label"].shape)
        break