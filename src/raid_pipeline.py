"""
RAID Dataset Pipeline
=====================
Downloads and prepares the RAID benchmark dataset (liamdugan/raid)
for training alongside or instead of HC3.

RAID includes 7.4M+ generations from 12 models across 8 domains.

Files:
  - train.csv: ~11.8 GB
  - test.csv:  ~1.2 GB
  - extra.csv: ~3.7 GB

Usage:
    python src/raid_pipeline.py                        # Download and prepare (full)
    python src/raid_pipeline.py --sample 50000         # Use subset for quick testing
    python src/raid_pipeline.py --merge-hc3            # Merge with existing HC3 data
    python src/raid_pipeline.py --skip-download        # Use existing local CSVs
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAID_RAW_DIR = DATA_DIR / "raid_raw"


def download_raid() -> None:
    """Download RAID CSVs from HuggingFace."""
    import requests

    RAID_RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "train.csv": "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv",
        "test.csv": "https://huggingface.co/datasets/liamdugan/raid/resolve/main/test.csv",
    }

    for fname, url in files.items():
        out_path = RAID_RAW_DIR / fname
        if out_path.exists():
            logger.info(f"  {fname} already exists, skipping download")
            continue

        logger.info(f"  Downloading {fname} (~{url.split('/')[-1]})...")
        logger.info(f"  URL: {url}")

        # Stream download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192 * 1024):  # 8MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / (1024 * 1024)
                    print(f"\r  {fname}: {mb:.0f} MB / {total/(1024*1024):.0f} MB ({pct:.1f}%)", end="", flush=True)
        print()
        logger.info(f"  Saved {fname} ({downloaded / (1024*1024):.0f} MB)")


def load_raid_chunked(max_samples: int | None = None, chunk_size: int = 100_000) -> pd.DataFrame:
    """Load RAID data using chunked reading for memory efficiency."""
    dfs = []

    for split in ["train", "test"]:
        csv_path = RAID_RAW_DIR / f"{split}.csv"
        if not csv_path.exists():
            logger.warning(f"  {csv_path} not found, skipping")
            continue

        logger.info(f"  Reading {split}.csv in chunks...")
        chunk_dfs = []

        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            # RAID columns: model, generation, domain, title, prompt, attack, etc.
            if "generation" not in chunk.columns:
                logger.warning(f"  Missing 'generation' column in chunk {i}")
                continue

            # Keep only what we need
            chunk = chunk[["model", "generation", "domain"]].copy()
            chunk.columns = ["raid_model", "text", "raid_domain"]

            # Filter invalid rows
            chunk = chunk.dropna(subset=["text"])
            chunk = chunk[chunk["text"].str.len() > 0]

            # Label: human=0, AI=1
            chunk["label"] = (chunk["raid_model"] != "human").astype(int)
            chunk["source"] = "raid"

            chunk_dfs.append(chunk)

            if max_samples and sum(len(d) for d in chunk_dfs) >= max_samples:
                break

        if chunk_dfs:
            split_df = pd.concat(chunk_dfs, ignore_index=True)
            if max_samples and len(split_df) > max_samples:
                split_df = split_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            dfs.append(split_df)
            logger.info(f"  {split}: {len(split_df)} samples")

    if not dfs:
        raise FileNotFoundError("No RAID data found. Run with --download first.")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total RAID: {len(df)} samples")
    logger.info(f"  Labels: {df['label'].value_counts().to_dict()}")
    logger.info(f"  Models: {df['raid_model'].value_counts().to_dict()}")
    logger.info(f"  Domains: {df['raid_domain'].value_counts().to_dict()}")
    return df


def clean_raid(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning consistent with HC3 pipeline."""
    logger.info("Cleaning RAID data...")
    before = len(df)

    df["text"] = df["text"].apply(lambda t: re.sub(r"\s+", " ", t).strip())
    df["text"] = df["text"].apply(lambda t: re.sub(r"https?://\S+|www\.\S+", "", t))

    df = df[df["text"].str.len() >= 50].reset_index(drop=True)
    df["text"] = df["text"].str[:2048]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    logger.info(f"  {before} -> {len(df)} samples after cleaning")
    logger.info(f"  human={len(df[df.label==0])}, ai={len(df[df.label==1])}")
    return df


def split_and_save(df: pd.DataFrame, name: str = "raid") -> None:
    """Split into train/val/test and save as CSV."""
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_val, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    val_relative = 0.1 / 0.8
    train, val = train_test_split(
        train_val, test_size=val_relative, random_state=42, stratify=train_val["label"]
    )

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}_{split_name}.csv"
        split_df.to_csv(path, index=False)
        n_human = len(split_df[split_df.label == 0])
        n_ai = len(split_df[split_df.label == 1])
        logger.info(f"  {split_name}: {len(split_df)} (human={n_human}, ai={n_ai}) -> {path}")


def merge_with_hc3() -> None:
    """Merge RAID splits with existing HC3 splits."""
    out_dir = DATA_DIR / "processed"

    for split_name in ["train", "val", "test"]:
        raid_path = out_dir / f"raid_{split_name}.csv"
        hc3_path = out_dir / f"{split_name}.csv"

        if not raid_path.exists() or not hc3_path.exists():
            logger.warning(f"Missing files for merge: {raid_path} or {hc3_path}")
            continue

        raid_df = pd.read_csv(raid_path)
        hc3_df = pd.read_csv(hc3_path)

        common = ["text", "label", "source"]
        merged = pd.concat([
            hc3_df[[c for c in common if c in hc3_df.columns]],
            raid_df[[c for c in common if c in raid_df.columns]],
        ], ignore_index=True)
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

        # Write to merged_*.csv rather than {split}.csv. Overwriting the HC3
        # splits in place would destroy the originals, drop their extra
        # columns, and make a second run merge RAID in twice.
        out_path = out_dir / f"merged_{split_name}.csv"
        merged.to_csv(out_path, index=False)
        logger.info(f"  Merged {split_name}: {len(merged)} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="RAID dataset pipeline")
    parser.add_argument("--download", action="store_true", help="Download RAID from HuggingFace")
    parser.add_argument("--sample", type=int, default=None, help="Max samples (None=all)")
    parser.add_argument("--merge-hc3", action="store_true", help="Merge with HC3 splits")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, use local CSVs")
    args = parser.parse_args()

    if not args.skip_download:
        logger.info("Step 1: Downloading RAID dataset...")
        download_raid()

    logger.info("Step 2: Loading and processing...")
    df = load_raid_chunked(max_samples=args.sample)

    logger.info("Step 3: Cleaning...")
    df = clean_raid(df)

    logger.info("Step 4: Splitting and saving...")
    split_and_save(df, name="raid")

    if args.merge_hc3:
        logger.info("Step 5: Merging with HC3...")
        merge_with_hc3()

    logger.info("Done!")
    logger.info(f"Files saved to: {DATA_DIR / 'processed'}")


if __name__ == "__main__":
    main()
