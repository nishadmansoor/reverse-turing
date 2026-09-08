"""
Shared Feature Extraction
=========================
Single source of truth for the stylometric features and the heatmap encoding.

Every training script, the evaluation harness, and the Flask app import from
here. Keeping one copy matters: if the app extracted features differently from
the script that trained the model, predictions would silently degrade.
"""

import string

import cv2
import numpy as np

# Order matters — it defines the column order of every feature vector and
# must stay in sync with the coefficients of any model trained on them.
FEATURE_NAMES = [
    "Avg Sentence Length",
    "Sentence Length Variance",
    "Avg Word Length",
    "Vocabulary Richness",
    "Punctuation Density",
]

N_FEATURES = len(FEATURE_NAMES)

IMAGE_SIZE = 224


def extract_features(text: str) -> list:
    """Extract the 5 stylometric features from a single text.

    Returns zeros for empty/unparseable text so callers never have to
    special-case it.
    """
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 0]
    words = text.lower().split()
    if len(sentences) == 0 or len(words) == 0:
        return [0.0] * N_FEATURES
    return [
        float(np.mean([len(s.split()) for s in sentences])),
        float(np.var([len(s.split()) for s in sentences])),
        float(np.mean([len(w) for w in words])),
        len(set(words)) / len(words),
        sum(1 for c in text if c in string.punctuation) / len(text),
    ]


def extract_features_batch(texts) -> np.ndarray:
    """Extract features for many texts. Returns an (n, N_FEATURES) array."""
    return np.array([extract_features(t) for t in texts], dtype=np.float64)


# Per-sentence measurements rendered as grid columns, each with a FIXED
# normalisation range. Fixed ranges are the whole point: they keep values
# comparable across texts, so "long words" looks the same in every sample.
HEATMAP_COLUMNS = [
    ("char_length", 300.0),
    ("word_count", 60.0),
    ("avg_word_length", 12.0),
    ("punctuation_count", 15.0),
    ("comma_count", 8.0),
    ("digit_count", 10.0),
    ("uppercase_ratio", 0.3),
    ("type_token_ratio", 1.0),
]

HEATMAP_ROWS = 64  # sentences kept (zero-padded / truncated)


def _sentence_row(sentence: str) -> list:
    words = sentence.split()
    n_words = len(words)
    n_chars = max(len(sentence), 1)
    return [
        len(sentence),
        n_words,
        sum(len(w) for w in words) / n_words if n_words else 0.0,
        sum(1 for c in sentence if c in string.punctuation),
        sentence.count(","),
        sum(1 for c in sentence if c.isdigit()),
        sum(1 for c in sentence if c.isupper()) / n_chars,
        len({w.lower() for w in words}) / n_words if n_words else 0.0,
    ]


def text_to_heatmap(text: str) -> np.ndarray:
    """Encode text as a 224x224x3 heatmap image for the CNN.

    Each sentence becomes a row of measurements; the grid is normalised
    against fixed global ranges, then nearest-neighbour resized and
    colour-mapped so a vision model reads structural rhythm as texture.

    Normalisation is deliberately global, not per-text. The original version
    applied `cv2.NORM_MINMAX` to each text individually, which rescaled every
    sample to the same 0-255 span and destroyed absolute magnitude — the exact
    signal the stylometric model relies on. That left the CNN with no class
    information (measured: identical mean P(AI) on both classes). See
    `text_to_heatmap_legacy` for the original behaviour.
    """
    rows = [_sentence_row(s) for s in text.split(".") if s.strip()]
    if not rows:
        rows = [[0.0] * len(HEATMAP_COLUMNS)]

    grid = np.array(rows[:HEATMAP_ROWS], dtype=np.float32)
    if len(grid) < HEATMAP_ROWS:
        grid = np.vstack([grid, np.zeros((HEATMAP_ROWS - len(grid), grid.shape[1]), np.float32)])

    scales = np.array([s for _, s in HEATMAP_COLUMNS], dtype=np.float32)
    grid = np.clip(grid / scales, 0.0, 1.0) * 255.0

    image = cv2.resize(grid.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE),
                       interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)


def text_to_heatmap_legacy(text: str) -> np.ndarray:
    """The original per-text min-max encoding. Kept to reproduce the
    negative result documented in the README."""
    rows = []
    for sentence in text.split("."):
        words = sentence.split()
        if len(words) > 0:
            rows.append([
                len(sentence),
                len(words),
                sum(len(w) for w in words) / len(words),
                sum(1 for c in sentence if c in string.punctuation),
            ])
    if not rows:
        rows = [[0, 0, 0, 0]]

    grid = np.array(rows, dtype=np.float32)
    grid = cv2.normalize(grid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.resize(grid, (IMAGE_SIZE, IMAGE_SIZE))
    return cv2.applyColorMap(image, cv2.COLORMAP_JET)


def meta_features(bert_ai_prob: float, stylo_ai_prob: float, cv_ai_prob: float,
                  text: str) -> np.ndarray:
    """Build the 5-column meta-classifier input.

    Column order is fixed by the trained meta-classifier:
        [bert_ai_prob, stylo_ai_prob, cv_ai_prob, word_count, sentence_count]
    Probabilities are 0-1, not percentages.
    """
    word_count = len(text.split())
    sentence_count = len([s for s in text.split(".") if s.strip()])
    return np.array([[
        bert_ai_prob,
        stylo_ai_prob,
        cv_ai_prob,
        word_count,
        sentence_count,
    ]], dtype=np.float64)


META_FEATURE_NAMES = [
    "bert_ai_prob",
    "stylo_ai_prob",
    "cv_ai_prob",
    "word_count",
    "sentence_count",
]
