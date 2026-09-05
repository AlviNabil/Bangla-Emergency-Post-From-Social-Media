"""Load the dataset splits from the Hugging Face Hub.

The splits are not kept in the repository. They are downloaded on first use and
cached under ~/.cache/huggingface, so training needs network access once per
machine and reads from disk after that.

Override the source repo with the DATASET_REPO environment variable.
"""

import csv
import os

from huggingface_hub import hf_hub_download

DATASET_REPO_ID = os.environ.get(
    "DATASET_REPO", "NightRaven/bangla-emergency-posts"
)

# alphabetical, matching the ids the trained models were fitted against
LABELS = ["accident", "blood", "crime", "fire", "natural_disaster",
          "pandemic", "suicide", "war", "weather"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def load_split(split):
    """Return one split as a list of [text, label_id], the shape BertDataset wants."""
    path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=f"data/{split}.csv",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8", newline="") as fh:
        return [[row["content"], LABEL_TO_ID[row["label"]]]
                for row in csv.DictReader(fh)]
