"""
Build Runtime Artifacts
=======================
Precomputes everything the Flask app needs at request time and writes it to
`artifacts/` as small files.

Why this exists
---------------
The app used to load `data/processed/train.csv` (195 MB) on every boot just to
refit the stylometric model and compute feature averages — about 15 s and well
over 500 MB resident per worker, which made deployment impossible and a clone
without the dataset unable to start at all.

Everything here is deterministic given the trained models, so it belongs in a
build step. The outputs total a few MB and are safe to commit, which is what
lets `git clone && python app/app.py` work.

Usage:
    python src/build_artifacts.py                    # HC3 (default)
    python src/build_artifacts.py --dataset raid     # RAID splits
    python src/build_artifacts.py --explorer-size 8000
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    FEATURE_NAMES,
    extract_features_batch,
    text_to_heatmap,
)
from src.model_defs import CNN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

STYLO_PATH = MODELS_DIR / "stylometric_classifier.pkl"
PCA_PATH = ARTIFACTS_DIR / "pca.pkl"
EXPLORER_PATH = ARTIFACTS_DIR / "explorer.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

BERT_BATCH = 32
CNN_BATCH = 64


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _split_paths(dataset: str) -> tuple:
    """Resolve train/test paths for a named dataset.

    Explicit rather than inferred — the previous prefix-probing loop always
    matched the HC3 files first, making the RAID branch unreachable.
    """
    prefix = "" if dataset == "hc3" else f"{dataset}_"
    train = DATA_DIR / f"{prefix}train.csv"
    test = DATA_DIR / f"{prefix}test.csv"
    missing = [p for p in (train, test) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split(s) for dataset '{dataset}': "
            + ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing)
        )
    return train, test


# ---------------------------------------------------------------------------
# Stylometric model (trained here, and actually saved)
# ---------------------------------------------------------------------------

def build_stylometric(train_df: pd.DataFrame) -> tuple:
    """Fit the stylometric classifier and the per-class feature averages.

    `retrain.py` trained this model but dropped the return value without
    saving, so it was silently refit from HC3 on every app boot.
    """
    logger.info("Extracting stylometric features for %d training rows...", len(train_df))
    X = extract_features_batch(train_df["text"])
    y = train_df["label"].to_numpy()

    logger.info("Fitting stylometric logistic regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    train_acc = float(model.score(X, y))
    logger.info("  train accuracy: %.4f", train_acc)

    averages = {
        "human": X[y == 0].mean(axis=0).tolist(),
        "ai": X[y == 1].mean(axis=0).tolist(),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(STYLO_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("  saved %s", STYLO_PATH.relative_to(PROJECT_ROOT))

    return model, averages, train_acc


# ---------------------------------------------------------------------------
# Batched inference over the explorer sample
# ---------------------------------------------------------------------------

def _softmax_ai(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1]


def _bert_ai_probs_onnx(texts) -> np.ndarray:
    """P(AI) from the quantized int8 graph the app actually serves."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_dir = MODELS_DIR / "onnx"
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir / "tokenizer"))
    sess = ort.InferenceSession(str(onnx_dir / "transformer_int8.onnx"),
                                providers=["CPUExecutionProvider"])
    logger.info("Running int8 transformer over %d explorer rows...", len(texts))

    out = []
    for start in range(0, len(texts), BERT_BATCH):
        batch = list(texts[start:start + BERT_BATCH])
        tokens = tokenizer(batch, max_length=256, padding="max_length",
                           truncation=True, return_tensors="np")
        logits = sess.run(None, {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
        })[0]
        out.extend(_softmax_ai(logits).tolist())
        if start % (BERT_BATCH * 20) == 0:
            logger.info("  %d/%d", start, len(texts))
    return np.array(out)


def _cnn_ai_probs_onnx(texts) -> np.ndarray:
    """P(AI) from the quantized CNN graph."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(MODELS_DIR / "onnx" / "cnn_int8.onnx"),
                                providers=["CPUExecutionProvider"])
    logger.info("Running int8 CNN over %d explorer rows...", len(texts))

    out = []
    for start in range(0, len(texts), CNN_BATCH):
        batch = [text_to_heatmap(t) for t in texts[start:start + CNN_BATCH]]
        images = np.stack(batch).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        out.extend(_softmax_ai(sess.run(None, {"images": images})[0]).tolist())
    return np.array(out)


def _bert_ai_probs(texts, bert_dir: Path) -> np.ndarray:
    """P(AI) for each text from the fine-tuned transformer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Running transformer over %d explorer rows...", len(texts))
    tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
    device = _device()
    model.to(device).eval()

    out = []
    for start in range(0, len(texts), BERT_BATCH):
        batch = list(texts[start:start + BERT_BATCH])
        tokens = tokenizer(batch, max_length=256, padding="max_length",
                           truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**tokens).logits, dim=1)[:, 1]
        out.extend(probs.cpu().tolist())
        if start % (BERT_BATCH * 20) == 0:
            logger.info("  %d/%d", start, len(texts))

    del model
    return np.array(out)


def _cnn_ai_probs(texts) -> np.ndarray:
    """P(AI) for each text from the heatmap CNN."""
    cnn_path = MODELS_DIR / "cnn_classifier.pt"
    logger.info("Running CNN over %d explorer rows...", len(texts))

    model = CNN()
    model.load_state_dict(torch.load(cnn_path, map_location="cpu", weights_only=True))
    device = _device()
    model.to(device).eval()

    out = []
    for start in range(0, len(texts), CNN_BATCH):
        batch = [text_to_heatmap(t) for t in texts[start:start + CNN_BATCH]]
        tensor = torch.tensor(np.stack(batch), dtype=torch.float32)
        tensor = tensor.permute(0, 3, 1, 2).to(device) / 255.0
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[:, 1]
        out.extend(probs.cpu().tolist())

    del model
    return np.array(out)


# ---------------------------------------------------------------------------
# Explorer payload
# ---------------------------------------------------------------------------

def build_explorer(test_df: pd.DataFrame, stylo_model, size: int, seed: int,
                   include_bert: bool, include_cnn: bool, backend: str = "onnx") -> tuple:
    """Build the PCA scatter payload for the Dataset Explorer.

    Predictions are computed from the models as they exist right now, over the
    exact sampled rows. The old code read months-old prediction CSVs and
    indexed them with a reset index, so every point was paired with the wrong
    model output and the reported accuracy was meaningless.
    """
    if len(test_df) > size:
        sample = test_df.sample(n=size, random_state=seed)
    else:
        sample = test_df
    sample = sample.reset_index(drop=True)
    texts = sample["text"].tolist()
    logger.info("Explorer sample: %d rows", len(sample))

    features = extract_features_batch(texts)

    # Standardise before PCA. The raw features differ by orders of magnitude
    # (sentence-length variance can reach the thousands while vocabulary
    # richness is bounded by 1), so unscaled PCA put 99.95% of variance on a
    # single component and collapsed the scatter plot into a line.
    pca = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=seed)),
    ])
    coords = pca.fit_transform(features)
    variance = pca.named_steps["pca"].explained_variance_ratio_
    logger.info("  PCA variance explained: %s", [round(float(v), 4) for v in variance])

    truth = sample["label"].to_numpy()
    stylo_ai = stylo_model.predict_proba(features)[:, 1]
    stylo_pred = (stylo_ai >= 0.5).astype(int)
    payload = {
        "x": [float(v) for v in coords[:, 0]],
        "y": [float(v) for v in coords[:, 1]],
        "true_label": [int(v) for v in truth],
        "stylo_pred": [int(v) for v in stylo_pred],
        "text_preview": [t[:120] for t in texts],
    }

    accuracies = {"stylometric": float((stylo_pred == truth).mean())}

    bert_dir = MODELS_DIR / "bert_classifier"
    has_bert = (bert_dir / "model.safetensors").exists() or (bert_dir / "pytorch_model.bin").exists()
    bert_ai = None
    if include_bert and (backend == "onnx" or has_bert):
        bert_ai = (_bert_ai_probs_onnx(texts) if backend == "onnx"
                   else _bert_ai_probs(texts, bert_dir))
        bert_pred = (bert_ai >= 0.5).astype(int)
        payload["bert_pred"] = [int(v) for v in bert_pred]
        accuracies["bert"] = float((bert_pred == truth).mean())
    else:
        logger.warning("Skipping transformer predictions for explorer")

    cnn_ai = None
    if include_cnn and (backend == "onnx" or (MODELS_DIR / "cnn_classifier.pt").exists()):
        cnn_ai = (_cnn_ai_probs_onnx(texts) if backend == "onnx"
                  else _cnn_ai_probs(texts))
        cnn_pred = (cnn_ai >= 0.5).astype(int)
        payload["cnn_pred"] = [int(v) for v in cnn_pred]
        accuracies["cnn"] = float((cnn_pred == truth).mean())
    else:
        logger.warning("Skipping CNN predictions for explorer")

    # Ensemble over the same sample, so every figure on the explorer page is
    # measured on identical rows.
    meta_path = (MODELS_DIR / "meta_classifier_int8.pkl" if backend == "onnx"
                 else MODELS_DIR / "meta_classifier.pkl")
    if bert_ai is not None and cnn_ai is not None and meta_path.exists():
        with open(meta_path, "rb") as f:
            meta_model = pickle.load(f)
        meta_X = np.column_stack([
            bert_ai,
            stylo_ai,
            cnn_ai,
            [len(t.split()) for t in texts],
            [len([s for s in t.split(".") if s.strip()]) for t in texts],
        ])
        ensemble_pred = (meta_model.predict_proba(meta_X)[:, 1] >= 0.5).astype(int)
        payload["ensemble_pred"] = [int(v) for v in ensemble_pred]
        accuracies["ensemble"] = float((ensemble_pred == truth).mean())
    else:
        logger.warning("Skipping ensemble predictions for explorer")

    for name, acc in accuracies.items():
        logger.info("  %s accuracy on explorer sample: %.4f", name, acc)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)
    with open(EXPLORER_PATH, "w") as f:
        json.dump(payload, f)
    logger.info("  saved %s (%.1f KB)", EXPLORER_PATH.relative_to(PROJECT_ROOT),
                EXPLORER_PATH.stat().st_size / 1024)

    return pca, accuracies, len(sample)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build runtime artifacts for the Flask app")
    parser.add_argument("--dataset", default="hc3", choices=["hc3", "raid", "merged"],
                        help="Which prepared splits to build from")
    parser.add_argument("--explorer-size", type=int, default=5000,
                        help="Rows to include in the Dataset Explorer scatter")
    parser.add_argument("--train-sample", type=int, default=None,
                        help="Subsample training rows before fitting (faster builds)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bert", action="store_true",
                        help="Skip transformer inference for the explorer")
    parser.add_argument("--skip-cnn", action="store_true",
                        help="Skip CNN inference for the explorer")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "torch"],
                        help="onnx (default) matches the models the app serves")
    args = parser.parse_args()

    train_path, test_path = _split_paths(args.dataset)

    logger.info("Loading %s", train_path.relative_to(PROJECT_ROOT))
    train_df = pd.read_csv(train_path).dropna(subset=["text"])
    if args.train_sample and len(train_df) > args.train_sample:
        train_df = train_df.sample(n=args.train_sample, random_state=args.seed)
        logger.info("  subsampled to %d rows", len(train_df))

    logger.info("Loading %s", test_path.relative_to(PROJECT_ROOT))
    test_df = pd.read_csv(test_path).dropna(subset=["text"])

    stylo_model, averages, stylo_train_acc = build_stylometric(train_df)

    pca, accuracies, explorer_n = build_explorer(
        test_df, stylo_model, args.explorer_size, args.seed,
        include_bert=not args.skip_bert, include_cnn=not args.skip_cnn,
        backend=args.backend,
    )

    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "backend": args.backend,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "explorer_rows": int(explorer_n),
        "feature_names": FEATURE_NAMES,
        "feature_averages": averages,
        "pca_variance_explained": [
            float(v) for v in pca.named_steps["pca"].explained_variance_ratio_
        ],
        "stylometric_train_accuracy": stylo_train_acc,
        "explorer_sample_accuracy": accuracies,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  saved %s", METADATA_PATH.relative_to(PROJECT_ROOT))

    logger.info("Done. Artifacts written to %s/", ARTIFACTS_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
