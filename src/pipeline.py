import re
import logging
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve()

def load_config(config_path: Optional[str] = None) -> dict:
    """Load YAML configuration."""
    if config_path is None:
        config_path = "/Users/nishad/Projects/reverse-turing/config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def preprocess_text(text: str, config: dict) -> str:
    prep = config.get("preprocessing", {})

    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # Strip extra whitespace
    if prep.get("strip_extra_whitespace", True):
        text = re.sub(r"\s+", " ", text).strip()

    # Remove URLs
    if prep.get("remove_urls", True):
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove special characters (disabled by default)
    if prep.get("remove_special_chars", False):
        text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", "", text)

    # Lowercase (disabled by default for BERT cased models)
    if prep.get("lowercase", False):
        text = text.lower()

    return text.strip()


def load_hc3(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    dataset_name = data_cfg["dataset_name"]
    subset = data_cfg.get("subset", "all")

    logger.info(f"Loading HC3 dataset: {dataset_name} (subset={subset})")
    ds = load_dataset(dataset_name, "default", revision="refs/convert/parquet")

    rows = []
    for split_name in ds:
        for item in ds[split_name]:
            question = item.get("question", "")

            # HC3 stores lists of human answers and chatgpt answers
            human_answers = item.get("human_answers", [])
            chatgpt_answers = item.get("chatgpt_answers", [])

            for answer in human_answers:
                rows.append({
                    "text": answer,
                    "label": 0,  # human
                    "source": subset,
                    "question": question,
                })

            for answer in chatgpt_answers:
                rows.append({
                    "text": answer,
                    "label": 1,  # ai
                    "source": subset,
                    "question": question,
                })

    df = pd.DataFrame(rows)
    logger.info(f"Raw HC3 samples: {len(df)} (human={len(df[df.label==0])}, ai={len(df[df.label==1])})")
    return df


def clean_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply preprocessing, length filtering, and optional sampling."""
    data_cfg = config["data"]

    # Preprocess text
    logger.info("Preprocessing text...")
    df = df.copy()
    df["text"] = df["text"].apply(lambda t: preprocess_text(t, config))

    # Drop empty texts
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Length filtering
    min_len = data_cfg.get("min_text_length", 50)
    max_len = data_cfg.get("max_text_length", 2048)

    before = len(df)
    df = df[df["text"].str.len() >= min_len].reset_index(drop=True)
    logger.info(f"Filtered by min_length ({min_len}): {before} -> {len(df)}")

    # Truncate long texts (keep full text in a separate column for reference)
    df["text_full"] = df["text"]
    df["text"] = df["text"].str[:max_len]

    # Optional: limit total samples (for quick dev iterations)
    max_samples = data_cfg.get("max_samples")
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=data_cfg["random_seed"])
        df = df.reset_index(drop=True)
        logger.info(f"Sampled down to {len(df)} samples")

    logger.info(f"Clean dataset: {len(df)} samples (human={len(df[df.label==0])}, ai={len(df[df.label==1])})")
    return df


def split_dataset(df: pd.DataFrame, config: dict) -> dict:
    data_cfg = config["data"]
    test_size = data_cfg.get("test_size", 0.2)
    val_size = data_cfg.get("val_size", 0.1)
    seed = data_cfg.get("random_seed", 42)

    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )

    # Second split: train vs val
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=seed, stratify=train_val["label"]
    )

    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    for name, split_df in splits.items():
        n_human = len(split_df[split_df.label == 0])
        n_ai = len(split_df[split_df.label == 1])
        logger.info(f"  {name}: {len(split_df)} samples (human={n_human}, ai={n_ai})")

    return splits


def save_splits(splits: dict, config: dict) -> None:
    """Save train/val/test splits to disk as CSV."""
    out_dir = PROJECT_ROOT / config["paths"]["processed_data"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in splits.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {name} split to {path}")


def load_splits(config: dict) -> dict:
    """Load previously saved splits from disk."""
    data_dir = PROJECT_ROOT / config["paths"]["processed_data"]
    splits = {}
    for name in ["train", "val", "test"]:
        path = data_dir / f"{name}.csv"
        if path.exists():
            splits[name] = pd.read_csv(path)
            logger.info(f"Loaded {name}: {len(splits[name])} samples")
        else:
            raise FileNotFoundError(f"Split file not found: {path}")
    return splits


def build_dataset(config: Optional[dict] = None, save: bool = True) -> dict:
    if config is None:
        config = load_config()

    df = load_hc3(config)
    df = clean_dataset(df, config)
    splits = split_dataset(df, config)

    if save:
        save_splits(splits, config)

    return splits


if __name__ == "__main__":
    config = load_config()
    splits = build_dataset(config, save=True)

    print("\n=== Dataset Summary ===")
    for name, df in splits.items():
        print(f"  {name}: {len(df)} samples | "
              f"human={len(df[df.label==0])} | "
              f"ai={len(df[df.label==1])}")
