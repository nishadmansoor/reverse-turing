"""
Train the Stacking Ensemble
===========================
Fits the meta-classifier that combines the three base models, then evaluates
everything on a held-out test split.

Protocol
--------
    base models  -> trained on  train split   (src/retrain.py)
    meta-classifier -> fitted on val split    (this script)
    all reported numbers -> measured on test split

Fitting the meta-classifier on the same data the base models saw would leak:
the bases are overconfident on their own training rows, so the stacker would
learn to trust them more than it should. Using the val split keeps the
combination honest.

This script exists because nothing in the repo could previously reproduce
`models/meta_classifier.pkl` — it was built by code that is no longer present,
and its coefficients no longer matched the models on disk.

Usage:
    python src/train_meta.py --dataset raid
    python src/train_meta.py --dataset raid --test-sample 10000
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    META_FEATURE_NAMES,
    extract_features_batch,
    text_to_heatmap,
)
from src.model_defs import CNN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

BERT_BATCH = 64
CNN_BATCH = 128


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Base model probabilities
# ---------------------------------------------------------------------------

def _softmax_ai(logits: np.ndarray) -> np.ndarray:
    """P(AI) column from raw logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True))[:, 1]


def bert_ai_probs(texts, device, backend="torch") -> np.ndarray:
    if backend == "onnx":
        return _bert_ai_probs_onnx(texts)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    bert_dir = MODELS_DIR / "bert_classifier"
    tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))
    model.to(device).eval()

    out = []
    for start in range(0, len(texts), BERT_BATCH):
        batch = list(texts[start:start + BERT_BATCH])
        tokens = tokenizer(batch, max_length=256, padding="max_length",
                           truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out.extend(torch.softmax(model(**tokens).logits, dim=1)[:, 1].cpu().tolist())
        if (start // BERT_BATCH) % 25 == 0:
            logger.info("    transformer %d/%d", start, len(texts))

    del model
    return np.array(out)


def _bert_ai_probs_onnx(texts) -> np.ndarray:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_dir = MODELS_DIR / "onnx"
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir / "tokenizer"))
    sess = ort.InferenceSession(str(onnx_dir / "transformer_int8.onnx"),
                                providers=["CPUExecutionProvider"])

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
        if (start // BERT_BATCH) % 25 == 0:
            logger.info("    transformer(int8) %d/%d", start, len(texts))

    return np.array(out)


def cnn_ai_probs(texts, device, backend="torch") -> np.ndarray:
    if backend == "onnx":
        return _cnn_ai_probs_onnx(texts)

    model = CNN()
    model.load_state_dict(torch.load(MODELS_DIR / "cnn_classifier.pt",
                                     map_location="cpu", weights_only=True))
    model.to(device).eval()

    out = []
    for start in range(0, len(texts), CNN_BATCH):
        batch = [text_to_heatmap(t) for t in texts[start:start + CNN_BATCH]]
        tensor = torch.from_numpy(np.stack(batch)).float().permute(0, 3, 1, 2).to(device) / 255.0
        with torch.no_grad():
            out.extend(torch.softmax(model(tensor), dim=1)[:, 1].cpu().tolist())

    del model
    return np.array(out)


def _cnn_ai_probs_onnx(texts) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(MODELS_DIR / "onnx" / "cnn_int8.onnx"),
                                providers=["CPUExecutionProvider"])
    out = []
    for start in range(0, len(texts), CNN_BATCH):
        batch = [text_to_heatmap(t) for t in texts[start:start + CNN_BATCH]]
        images = np.stack(batch).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        logits = sess.run(None, {"images": images})[0]
        out.extend(_softmax_ai(logits).tolist())

    return np.array(out)


def stylo_ai_probs(texts) -> np.ndarray:
    with open(MODELS_DIR / "stylometric_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict_proba(extract_features_batch(texts))[:, 1]


def build_meta_frame(df: pd.DataFrame, device, label: str, backend="torch") -> pd.DataFrame:
    """Run all three base models over a split and assemble meta-features."""
    texts = df["text"].tolist()
    logger.info("  [%s] %d rows (backend=%s)", label, len(texts), backend)

    logger.info("  [%s] stylometric...", label)
    stylo = stylo_ai_probs(texts)
    logger.info("  [%s] cnn...", label)
    cnn = cnn_ai_probs(texts, device, backend)
    logger.info("  [%s] transformer...", label)
    bert = bert_ai_probs(texts, device, backend)

    frame = pd.DataFrame({
        "bert_ai_prob": bert,
        "stylo_ai_prob": stylo,
        "cv_ai_prob": cnn,
        "word_count": [len(t.split()) for t in texts],
        "sentence_count": [len([s for s in t.split(".") if s.strip()]) for t in texts],
        "label": df["label"].to_numpy(),
    })
    return frame[META_FEATURE_NAMES + ["label"]]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score(truth, pred, prob=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, pred).tolist(),
        "majority_baseline": float(max(np.mean(truth), 1 - np.mean(truth))),
    }
    if prob is not None:
        out["roc_auc"] = float(roc_auc_score(truth, prob))
    return out


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate the stacking ensemble")
    parser.add_argument("--dataset", default="raid", choices=["hc3", "raid", "merged"])
    parser.add_argument("--val-sample", type=int, default=None,
                        help="Subsample the val split used to fit the stacker")
    parser.add_argument("--test-sample", type=int, default=None,
                        help="Subsample the test split used for reporting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="torch", choices=["torch", "onnx"],
                        help="onnx uses the quantized int8 graphs in models/onnx/")
    args = parser.parse_args()

    # Keep the two backends' outputs side by side so the quantization delta is
    # measurable rather than assumed.
    suffix = "" if args.backend == "torch" else "_int8"

    prefix = "" if args.dataset == "hc3" else f"{args.dataset}_"
    val_path = DATA_DIR / f"{prefix}val.csv"
    test_path = DATA_DIR / f"{prefix}test.csv"
    for path in (val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.relative_to(PROJECT_ROOT)}")

    device = get_device()
    logger.info("Device: %s", device)

    val_df = pd.read_csv(val_path).dropna(subset=["text"])
    test_df = pd.read_csv(test_path).dropna(subset=["text"])
    if args.val_sample and len(val_df) > args.val_sample:
        val_df = val_df.sample(n=args.val_sample, random_state=args.seed)
    if args.test_sample and len(test_df) > args.test_sample:
        test_df = test_df.sample(n=args.test_sample, random_state=args.seed)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info("Building meta-features...")
    val_meta = build_meta_frame(val_df, device, "val", args.backend)
    test_meta = build_meta_frame(test_df, device, "test", args.backend)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    val_meta.to_csv(RESULTS_DIR / f"meta_features_val_{args.dataset}{suffix}.csv", index=False)
    test_meta.to_csv(RESULTS_DIR / f"meta_features_test_{args.dataset}{suffix}.csv", index=False)

    # --- Fit the stacker on val ---
    X_val = val_meta[META_FEATURE_NAMES].to_numpy()
    y_val = val_meta["label"].to_numpy()
    logger.info("Fitting meta-classifier on %d val rows...", len(X_val))
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta.fit(X_val, y_val)
    logger.info("  coefficients: %s",
                {n: round(float(c), 4) for n, c in zip(META_FEATURE_NAMES, meta.coef_[0])})

    meta_path = MODELS_DIR / f"meta_classifier{suffix}.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    logger.info("  saved %s", meta_path.relative_to(PROJECT_ROOT))

    # --- Evaluate everything on test ---
    y_test = test_meta["label"].to_numpy()
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "backend": args.backend,
        "val_rows_for_stacker": int(len(X_val)),
        "test_rows": int(len(y_test)),
        "meta_coefficients": {n: float(c) for n, c in zip(META_FEATURE_NAMES, meta.coef_[0])},
        "models": {},
    }

    for name, column in [("transformer", "bert_ai_prob"),
                         ("stylometric", "stylo_ai_prob"),
                         ("cnn", "cv_ai_prob")]:
        prob = test_meta[column].to_numpy()
        report["models"][name] = score(y_test, (prob >= 0.5).astype(int), prob)

    ensemble_prob = meta.predict_proba(test_meta[META_FEATURE_NAMES].to_numpy())[:, 1]
    report["models"]["ensemble"] = score(y_test, (ensemble_prob >= 0.5).astype(int), ensemble_prob)

    report_path = RESULTS_DIR / f"evaluation_{args.dataset}{suffix}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("")
    logger.info("=== Test results (%s, n=%d) ===", args.dataset, len(y_test))
    logger.info("%-14s %9s %9s %9s %9s", "model", "accuracy", "f1", "roc_auc", "vs base")
    baseline = report["models"]["ensemble"]["majority_baseline"]
    for name, metrics in report["models"].items():
        logger.info("%-14s %9.4f %9.4f %9.4f %+9.4f", name, metrics["accuracy"],
                    metrics["f1"], metrics.get("roc_auc", float("nan")),
                    metrics["accuracy"] - baseline)
    logger.info("majority baseline = %.4f", baseline)
    logger.info("")
    logger.info("Full report: %s", report_path.relative_to(PROJECT_ROOT))
    logger.info("Next: python src/build_artifacts.py --dataset %s", args.dataset)

    print()
    print(classification_report(y_test, (ensemble_prob >= 0.5).astype(int),
                               target_names=["Human", "AI"], digits=4))


if __name__ == "__main__":
    main()
