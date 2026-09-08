"""
Train / Retrain Models
======================
Trains the transformer, stylometric, and CNN classifiers on a named dataset.

Usage:
    python src/retrain.py --dataset raid                  # all three
    python src/retrain.py --dataset raid --only bert      # one model
    python src/retrain.py --dataset raid --only cnn --cnn-epochs 8
    python src/retrain.py --dataset hc3 --max-steps 50    # smoke test

Notes on prior failure modes this script guards against
-------------------------------------------------------
* Dataset selection is an explicit flag. The previous prefix-probing loop
  (`for prefix in ["", "raid_"]`) always matched the HC3 files first, so the
  RAID branch was unreachable and `--dataset raid` runs silently trained on HC3.
* The stylometric model is now saved. It used to be fitted and discarded.
* Heatmaps are generated lazily per batch. Materialising all 111k RAID images
  up front needs ~17 GB and would OOM before the first epoch.
* Every run reports how far the transformer weights moved from their
  pretrained initialisation, and checkpoints each epoch. A previous run
  finished "successfully" having drifted only 4.7e-4 — effectively untrained —
  and that went unnoticed until the model was evaluated.
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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import extract_features_batch, text_to_heatmap  # noqa: E402
from src.model_defs import CNN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

BASE_TRANSFORMER = "distilbert-base-uncased"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_splits(dataset: str, sample: int | None, seed: int) -> tuple:
    """Load train/val splits for an explicitly named dataset."""
    prefix = "" if dataset == "hc3" else f"{dataset}_"
    paths = {s: DATA_DIR / f"{prefix}{s}.csv" for s in ("train", "val")}

    missing = [str(p.relative_to(PROJECT_ROOT)) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset '{dataset}' is missing: {', '.join(missing)}\n"
            "  For RAID: python src/raid_pipeline.py --download\n"
            "  For HC3:  python src/pipeline.py"
        )

    out = {}
    for split, path in paths.items():
        df = pd.read_csv(path).dropna(subset=["text"])
        if sample and len(df) > sample:
            # Keep val proportionally smaller so the ratio stays sane.
            n = sample if split == "train" else max(1000, sample // 8)
            df = df.sample(n=min(n, len(df)), random_state=seed)
        out[split] = df.reset_index(drop=True)
        logger.info("  %-5s %6d rows (human=%d, ai=%d) from %s",
                    split, len(out[split]),
                    int((out[split].label == 0).sum()), int((out[split].label == 1).sum()),
                    path.name)
    return out["train"], out["val"]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Tokenises on access so we never hold the whole tokenised corpus."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token = self.tokenizer(
            self.texts[idx], max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": token["input_ids"].squeeze(0),
            "attention_mask": token["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.labels[idx])),
        }


class HeatmapDataset(Dataset):
    """Renders each heatmap on access.

    Eager rendering of RAID's 111k rows at 224x224x3 needs roughly 17 GB.
    """

    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image = text_to_heatmap(self.texts[idx])
        tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        return tensor, torch.tensor(int(self.labels[idx]))


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

def _weight_drift(model, base_state) -> float:
    """Mean absolute change in body weights since initialisation."""
    current = model.state_dict()
    deltas = [
        (current[k].detach().float().cpu() - base_state[k].float()).abs().mean().item()
        for k in base_state
        if "classifier" not in k and current[k].shape == base_state[k].shape
    ]
    return float(np.mean(deltas)) if deltas else 0.0


def train_transformer(train_df, val_df, epochs, lr, batch_size, max_steps, out_dir) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Training transformer (%s)...", BASE_TRANSFORMER)
    device = get_device()
    logger.info("  device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(BASE_TRANSFORMER)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_TRANSFORMER, num_labels=2)
    model.to(device)

    base_state = {k: v.detach().float().cpu().clone()
                  for k, v in model.state_dict().items() if "classifier" not in k}

    optimizer = AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(TextDataset(train_df.text, train_df.label, tokenizer),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_df.text, val_df.label, tokenizer),
                            batch_size=batch_size * 2, shuffle=False)

    total_steps = len(train_loader) if not max_steps else min(max_steps, len(train_loader))
    logger.info("  %d steps/epoch x %d epochs", total_steps, epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_acc = [], -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(tqdm(train_loader, total=total_steps,
                                          desc=f"epoch {epoch}/{epochs}")):
            if max_steps and step >= max_steps:
                break
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += out.loss.item()

        preds, truth = _eval_transformer(model, val_loader, device)
        acc = accuracy_score(truth, preds)
        f1 = f1_score(truth, preds)
        drift = _weight_drift(model, base_state)
        logger.info("  epoch %d: loss=%.4f val_acc=%.4f val_f1=%.4f weight_drift=%.3e",
                    epoch, running / max(total_steps, 1), acc, f1, drift)
        history.append({"epoch": epoch, "val_accuracy": acc, "val_f1": f1, "weight_drift": drift})

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            logger.info("    new best — checkpointed to %s", out_dir.relative_to(PROJECT_ROOT))

    final_drift = history[-1]["weight_drift"]
    if final_drift < 1e-3:
        logger.warning(
            "Weight drift is only %.3e — the model is barely fine-tuned and will "
            "not be usable. Check that steps actually ran.", final_drift
        )

    return {"best_val_accuracy": best_acc, "history": history, "final_drift": final_drift}


def _eval_transformer(model, loader, device) -> tuple:
    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            truth.extend(batch["label"].tolist())
    return preds, truth


# ---------------------------------------------------------------------------
# Stylometric
# ---------------------------------------------------------------------------

def train_stylometric(train_df, val_df) -> dict:
    logger.info("Training stylometric model...")
    X_train = extract_features_batch(train_df.text)
    X_val = extract_features_batch(val_df.text)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_df.label.to_numpy())

    train_acc = float(model.score(X_train, train_df.label.to_numpy()))
    val_acc = float(model.score(X_val, val_df.label.to_numpy()))
    logger.info("  train_acc=%.4f val_acc=%.4f", train_acc, val_acc)

    out_path = MODELS_DIR / "stylometric_classifier.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("  saved %s", out_path.relative_to(PROJECT_ROOT))

    return {"train_accuracy": train_acc, "val_accuracy": val_acc}


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

def train_cnn(train_df, val_df, epochs, batch_size, lr, workers) -> dict:
    logger.info("Training CNN on text heatmaps...")
    device = get_device()
    logger.info("  device: %s", device)

    train_loader = DataLoader(HeatmapDataset(train_df.text, train_df.label),
                              batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(HeatmapDataset(val_df.text, val_df.label),
                            batch_size=batch_size, shuffle=False, num_workers=workers)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    out_path = MODELS_DIR / "cnn_classifier.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    history, best_acc = [], -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        preds, truth = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                truth.extend(labels.tolist())

        acc = accuracy_score(truth, preds)
        f1 = f1_score(truth, preds)
        logger.info("  epoch %d: loss=%.4f val_acc=%.4f val_f1=%.4f",
                    epoch, running / len(train_loader), acc, f1)
        history.append({"epoch": epoch, "val_accuracy": acc, "val_f1": f1})

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), str(out_path))
            logger.info("    new best — checkpointed to %s", out_path.relative_to(PROJECT_ROOT))

    return {"best_val_accuracy": best_acc, "history": history}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train models on a named dataset")
    parser.add_argument("--dataset", default="raid", choices=["hc3", "raid", "merged"])
    parser.add_argument("--only", nargs="*", choices=["bert", "stylo", "cnn"],
                        help="Train only these (default: all)")
    parser.add_argument("--epochs", type=int, default=3, help="Transformer epochs")
    parser.add_argument("--cnn-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cnn-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cnn-lr", type=float, default=1e-3)
    parser.add_argument("--sample", type=int, default=None, help="Subsample training rows")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap steps/epoch (smoke tests)")
    # Default 0: worker processes need shared-memory IPC via torch_shm_manager,
    # which is blocked in some sandboxed/containerised shells ("execl failed:
    # Permission denied"). Heatmap rendering is fast enough inline.
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers for CNN")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    targets = args.only or ["bert", "stylo", "cnn"]
    logger.info("Dataset: %s | training: %s", args.dataset, ", ".join(targets))

    train_df, val_df = load_splits(args.dataset, args.sample, args.seed)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "models": {},
    }

    if "stylo" in targets:
        report["models"]["stylometric"] = train_stylometric(train_df, val_df)

    if "cnn" in targets:
        report["models"]["cnn"] = train_cnn(
            train_df, val_df, args.cnn_epochs, args.cnn_batch_size, args.cnn_lr, args.workers
        )

    if "bert" in targets:
        report["models"]["transformer"] = train_transformer(
            train_df, val_df, args.epochs, args.lr, args.batch_size,
            args.max_steps, MODELS_DIR / "bert_classifier",
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"training_report_{args.dataset}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written to %s", report_path.relative_to(PROJECT_ROOT))

    logger.info("Done. Next: python src/train_meta.py --dataset %s", args.dataset)


if __name__ == "__main__":
    main()
