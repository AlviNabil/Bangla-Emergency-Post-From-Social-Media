import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "ModelTraining"))

from models.text_xlm_roberta import TextRoBERTa
from models.bangla_bert2 import BanglaBERT2
from models.text_bert import TextBERT
from bert_dataset import BertDataset
import torch
import numpy as np
from functools import lru_cache
import torch.utils.data

from model_store import model_file

_SPECS = {
    "BanglaBERT": (
        "sagorsarker/bangla-bert-base",
        BanglaBERT2,
        "Transformer/BanglaBERT/social-media_weights.pt",
    ),
    "XLM-RoBERTa": (
        "xlm-roberta-base",
        TextRoBERTa,
        "Transformer/XLM-RoBERTa/social-media_weights.pt",
    ),
    "mBERT": (
        "bert-base-multilingual-uncased",
        TextBERT,
        "Transformer/mBERT/social-media_weights.pt",
    ),
}


@lru_cache(maxsize=len(_SPECS))
def load_model(model_):
    """Build the wrapper and load its fine-tuned weights, once per process.

    Each checkpoint is 600 MB-1 GB, so rebuilding it on every prediction makes
    the app feel broken. lru_cache keeps one instance per model name.
    """
    pretrained, cls, weights = _SPECS[model_]
    model = cls(pretrained_model=pretrained, num_class=9, fine_tune=True)
    model.load_state_dict(
        torch.load(model_file(weights), map_location=torch.device("cpu"))
    )
    model.eval()
    return model, pretrained


def embedding(text, model_):
    model, pretrained = load_model(model_)
    text = [[text[0], 5]]
    text_dataset = BertDataset(text, 100, pretrained, 9, False)
    text_loader = torch.utils.data.DataLoader(
        text_dataset, batch_size=32, shuffle=False
    )

    criterion = torch.nn.CrossEntropyLoss()
    device = "cpu"
    model.eval()
    test_loss = 0
    total = 0
    correct = 0

    y_true_all = np.zeros((0,), dtype=int)
    y_pred_all = np.zeros((0,), dtype=int)

    with torch.no_grad():
        for x, att, y in text_loader:
            print(device)
            x, y, att = x.to(device), y.to(device), att.to(device)
            y_pred = model(x, att)

            loss = criterion(y_pred, y)

            test_loss += loss.item() * y.shape[0]
            total += y.shape[0]
            correct += torch.sum(torch.argmax(y_pred, dim=1) == y).item()

            y_true_all = np.concatenate([y_true_all, y.cpu().detach().numpy()])
            y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
            y_pred_all = np.concatenate([y_pred_all, y_pred])
    return y_pred_all
