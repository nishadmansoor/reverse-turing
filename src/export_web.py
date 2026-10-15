"""
Export Model Parameters and Parity Fixtures for the Browser Build
=================================================================
The static site cannot run scikit-learn, so the three linear pieces — the
stylometric classifier, the stacking meta-classifier, and the PCA projection
used by the Dataset Explorer — are exported as plain numbers for the JavaScript
port to consume.

It also writes a parity fixture: a set of texts with the exact outputs the
Python pipeline produces for them. `docs/parity.html` replays that fixture in
the browser and fails loudly on any mismatch, so a reimplementation that drifts
from the original cannot ship quietly.

Usage:
    python src/export_web.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    FEATURE_NAMES,
    HEATMAP_COLUMNS,
    HEATMAP_ROWS,
    extract_features,
    text_to_heatmap,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WEB_DIR = PROJECT_ROOT / "docs"
OUT_DIR = WEB_DIR / "data"


def _lr(path: Path) -> dict:
    """Logistic regression reduces to coefficients plus an intercept."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return {
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "classes": model.classes_.tolist(),
    }


def _pca(path: Path) -> dict:
    """The explorer's projection is a StandardScaler followed by a PCA."""
    with open(path, "rb") as f:
        pipe = pickle.load(f)
    scaler = pipe.named_steps["scale"]
    pca = pipe.named_steps["pca"]
    return {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def build_fixture(n_dataset_rows: int = 12) -> list:
    """Texts plus the exact numbers Python produces for them.

    Deliberately includes awkward cases — empty-ish input, no sentence breaks,
    heavy punctuation, unicode, very long text — because those are where a
    hand-written JavaScript port is most likely to diverge.
    """
    edge_cases = [
        "Short sentence here.",
        "no punctuation at all just words running on and on without any stops",
        "Multiple!!! Exclamations??? And... ellipses... everywhere!!!",
        "Café naïve résumé — unicode, em-dashes, and “curly quotes”.",
        "A. B. C. D. E. F. G.",
        "1234567890 42 3.14159 2026 100%",
        "   leading and trailing whitespace   ",
        "One.Two.Three.NoSpaces.AfterPeriods.",
        "x " * 300,
    ]

    rows = list(edge_cases)
    test_path = PROJECT_ROOT / "data" / "processed" / "raid_test.csv"
    if test_path.exists():
        sample = pd.read_csv(test_path).dropna(subset=["text"])
        sample = sample.sample(n=n_dataset_rows, random_state=99)
        rows.extend(sample["text"].tolist())
        logger.info("  added %d rows from the RAID test split", len(sample))
    else:
        logger.warning("  %s missing — fixture uses edge cases only", test_path)

    examples = json.loads((PROJECT_ROOT / "app" / "examples.json").read_text())
    rows.extend(e["text"] for e in examples)

    fixture = []
    for text in rows:
        features = extract_features(text)
        heatmap = text_to_heatmap(text)
        fixture.append({
            "text": text,
            "features": features,
            # Full heatmaps would bloat the fixture; a checksum and a few
            # sampled pixels pin the encoding down just as well.
            "heatmap_sum": int(heatmap.sum()),
            "heatmap_samples": [
                int(heatmap[r, c, ch])
                for r, c, ch in [(0, 0, 0), (0, 0, 2), (50, 50, 1),
                                 (112, 112, 0), (223, 223, 2)]
            ],
            "word_count": len(text.split()),
            "sentence_count": len([s for s in text.split(".") if s.strip()]),
        })
    return fixture


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting linear model parameters...")

    # OpenCV's JET colormap is a specific 256-entry BGR table. Exporting it
    # verbatim removes any chance of the JavaScript port approximating it and
    # feeding the CNN subtly different pixels than it was trained on.
    import cv2
    jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1),
                            cv2.COLORMAP_JET).reshape(256, 3)

    params = {
        "stylometric": _lr(MODELS_DIR / "stylometric_classifier.pkl"),
        "meta": _lr(MODELS_DIR / "meta_classifier_int8.pkl"),
        "pca": _pca(ARTIFACTS_DIR / "pca.pkl"),
        "feature_names": FEATURE_NAMES,
        "heatmap": {
            "rows": HEATMAP_ROWS,
            "size": 224,
            "columns": [name for name, _ in HEATMAP_COLUMNS],
            "scales": [scale for _, scale in HEATMAP_COLUMNS],
            # BGR, matching cv2.applyColorMap's output channel order.
            "jet_bgr": jet.tolist(),
        },
        # string.punctuation — the exact set the Python feature extractor counts.
        "punctuation": "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    }
    (OUT_DIR / "params.json").write_text(json.dumps(params, indent=2))
    logger.info("  wrote %s", (OUT_DIR / "params.json").relative_to(PROJECT_ROOT))
    logger.info("  stylometric coef: %s",
                [round(c, 4) for c in params["stylometric"]["coef"]])
    logger.info("  meta coef:        %s",
                [round(c, 4) for c in params["meta"]["coef"]])

    logger.info("Building parity fixture...")
    fixture = build_fixture()
    (OUT_DIR / "parity.json").write_text(json.dumps(fixture))
    logger.info("  wrote %d cases to %s",
                len(fixture), (OUT_DIR / "parity.json").relative_to(PROJECT_ROOT))

    # Metadata and the explorer payload carry over unchanged.
    for name in ("metadata.json", "explorer.json"):
        src = ARTIFACTS_DIR / name
        if src.exists():
            (OUT_DIR / name).write_text(src.read_text())
            logger.info("  copied %s (%.0f KB)", name, src.stat().st_size / 1024)

    # The Method & Results page reads this directly, so the published figures
    # always come from the evaluation run rather than being typed by hand.
    dataset = json.loads((ARTIFACTS_DIR / "metadata.json").read_text()).get("dataset", "raid")
    report = PROJECT_ROOT / "results" / f"evaluation_{dataset}_int8.json"
    if report.exists():
        (OUT_DIR / "evaluation.json").write_text(report.read_text())
        logger.info("  copied %s -> evaluation.json", report.name)
    else:
        logger.warning("  no evaluation report at %s", report.relative_to(PROJECT_ROOT))

    examples = (PROJECT_ROOT / "app" / "examples.json").read_text()
    (OUT_DIR / "examples.json").write_text(examples)
    logger.info("  copied examples.json")

    logger.info("Done. Next: port the JS and open docs/parity.html")


if __name__ == "__main__":
    main()
