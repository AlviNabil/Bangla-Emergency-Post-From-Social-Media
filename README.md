# Bangla Emergency Post Classification

Classifying Bangla social media posts into nine emergency categories, so that
posts needing attention from emergency services, local authorities or law
enforcement can be surfaced automatically.

From _Bangla Emergency Post Classification on Social Media using Transformer
Based BERT Models_, 6th International Conference on Electrical Information and
Communication Technology (EICT), Khulna, Bangladesh, December 2023.

```
Inference_UI/      Streamlit app - type a post, get a predicted category
ModelTraining/    training code for the transformer models
```

Neither the trained weights nor the dataset live in this repository. Both are
pulled from the Hugging Face Hub on first use and cached under
`~/.cache/huggingface`:

|                  |                                                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| models (3.3 GB)  | [NightRaven/bangla-emergency-post-classification](https://huggingface.co/NightRaven/bangla-emergency-post-classification) |
| dataset (2.6 MB) | [NightRaven/bangla-emergency-posts](https://huggingface.co/datasets/NightRaven/bangla-emergency-posts)                    |

So a clone is small, but **the first run downloads several GB** and needs
network access.

## Requirements

**Python 3.10 specifically.** TensorFlow 2.10 and torch 1.12.1 publish no wheels
for 3.11+, and pip will fail to resolve them there.

```bash
python3.10 -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows the activate line is `.venv\Scripts\activate`.

## Running the app

```bash
pip install -r Inference_UI/requirements.txt
```

```bash
streamlit run Inference_UI/app.py
```

Opens on <http://localhost:8501>. Pick an approach and a model, paste a Bangla
post, press Predict. The first use of each model downloads it (up to ~1 GB), so
that click can take several minutes; afterwards it is cached and instant.

A GPU is optional here — inference runs fine on CPU.

## Training

```bash
pip install -r ModelTraining/requirements.txt
```

```bash
cd ModelTraining && python train.py --model=<model-name> --task=social-media --batch-size=16 --lr=0.00001 --epoch=10 --save-path=<output-path>
```

for example:

```bash
cd ModelTraining && python train.py --model=xlm-roberta-base --task=social-media --batch-size=16 --lr=0.00001 --epoch=10 --save-path=model-weights
```

Outputs land in `ModelTraining/logs/<save-path>/`: `social-media_weights.pt` (best
checkpoint by validation accuracy), `social-media_logs.csv` (per-epoch metrics)
and `social-media_result.txt` (final test metrics and confusion matrix).

**A GPU is effectively required for training.** The default pin installs the CPU
build of torch; see the note at the bottom of `ModelTraining/requirements.txt` for the
CUDA wheel. Training XLM-RoBERTa for 10 epochs takes roughly half an hour on a
GTX 1070 Ti and impractically long on CPU.

To serve a newly trained model, upload it over the Hub copy and clear the cached
version so it is re-fetched:

```bash
hf upload NightRaven/bangla-emergency-post-classification ModelTraining/logs/social-media/xlm-base/social-media_weights.pt Transformer/XLM-RoBERTa/social-media_weights.pt --repo-type model
```

## Labels

Nine classes, encoded alphabetically:

| id | label    | id | label            | id | label   |
| -- | -------- | -- | ---------------- | -- | ------- |
| 0  | accident | 3  | fire             | 6  | suicide |
| 1  | blood    | 4  | natural_disaster | 7  | war     |
| 2  | crime    | 5  | pandemic         | 8  | weather |

5,836 posts, split 56 / 14 / 30 into 3,267 train, 819 validation, 1,750 test.
The classes are heavily imbalanced — `crime` is 42.7%, `pandemic` 2.5%.

## Attribution

```bibtex
@inproceedings{nabil2023bangla,
  title     = {Bangla Emergency Post Classification on Social Media using
               Transformer Based BERT Models},
  author    = {Nabil, Alvi Ahmmed and Arifeen, Shamsul and Das, Dola and
               Salim, Md. Shahidul and Fattah, H. M. Abdul},
  booktitle = {6th International Conference on Electrical Information and
               Communication Technology (EICT)},
  address   = {Khulna, Bangladesh},
  year      = {2023}
}
```
