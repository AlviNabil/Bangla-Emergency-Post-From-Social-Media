"""Resolve model files, preferring a local copy and falling back to the Hub.

The trained weights are too large for git (3.3 GB, six files over GitHub's
100 MB per-file limit), so they live in a Hugging Face model repo instead.
A checkout that already has Inference_UI/models/ keeps using it; anywhere else the files
are fetched once and cached under ~/.cache/huggingface.

Override the source repo with the MODELS_REPO environment variable.
"""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = os.environ.get(
    "MODELS_REPO", "NightRaven/bangla-emergency-post-classification"
)
DATASET_REPO_ID = os.environ.get("DATASET_REPO", "NightRaven/bangla-emergency-posts")
LOCAL_MODELS = Path(__file__).resolve().parent / "models"
LOCAL_DATA = Path(__file__).resolve().parent / "data"


def model_file(rel_path):
    """Absolute path to a single model file (a .pt or .pkl)."""
    local = LOCAL_MODELS / rel_path
    if local.is_file():
        return str(local)
    return hf_hub_download(repo_id=REPO_ID, filename=str(rel_path).replace("\\", "/"))


def model_dir(rel_path):
    """Absolute path to a TensorFlow SavedModel directory.

    SavedModels are several files, so this pulls the whole subtree rather than
    a single blob.
    """
    local = LOCAL_MODELS / rel_path
    if local.is_dir():
        return str(local)
    rel = str(rel_path).replace("\\", "/")
    root = snapshot_download(repo_id=REPO_ID, allow_patterns=f"{rel}/*")
    return str(Path(root) / rel)


def dataset_file(rel_path):
    """Absolute path to a file from the dataset repo.

    tokenizer.py fits its Keras vocabulary and TF-IDF space on the training
    split, so the app needs that CSV even though it never trains anything.
    """
    local = LOCAL_DATA / rel_path
    if local.is_file():
        return str(local)
    return hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=str(rel_path).replace("\\", "/"),
        repo_type="dataset",
    )
