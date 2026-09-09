"""
Export and Quantize Models for Deployment
=========================================
Converts the trained PyTorch models to int8 ONNX so the serving container can
drop the torch dependency entirely.

Why
---
The fp32 deployment needs ~1.1 GB resident (268 MB transformer weights + 25 MB
CNN + torch's runtime), which exceeds every free container tier. Dynamic int8
quantization plus onnxruntime brings that to roughly 130 MB.

Both models must be converted. Leaving either on torch keeps the ~800 MB
dependency and defeats the purpose.

Accuracy is not assumed — run `src/evaluate_onnx.py` afterwards to measure the
delta against the fp32 baseline on the held-out test split.

Usage:
    python src/quantize.py
    python src/quantize.py --skip-cnn
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import IMAGE_SIZE  # noqa: E402
from src.model_defs import CNN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"

OPSET = 14
MAX_TOKENS = 256


def _mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def export_transformer() -> None:
    """Export DistilBERT to ONNX, then dynamically quantize to int8."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src_dir = MODELS_DIR / "bert_classifier"
    fp32_path = ONNX_DIR / "transformer_fp32.onnx"
    int8_path = ONNX_DIR / "transformer_int8.onnx"

    logger.info("Exporting transformer to ONNX...")
    tokenizer = AutoTokenizer.from_pretrained(str(src_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(src_dir))
    model.eval()

    dummy = tokenizer(
        "example input for tracing the graph",
        max_length=MAX_TOKENS, padding="max_length",
        truncation=True, return_tensors="pt",
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        # Batch is dynamic so the server can size batches freely; sequence
        # length is fixed at MAX_TOKENS because the app always pads to it.
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
        # Legacy TorchScript exporter. torch>=2.9 defaults to the dynamo path,
        # whose graphs fail onnxruntime's shape inference during quantization
        # ("Inferred shape and existing shape differ").
        dynamo=False,
    )
    logger.info("  fp32 ONNX: %.1f MB", _mb(fp32_path))

    logger.info("  quantizing to int8...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    logger.info("  int8 ONNX: %.1f MB  (%.1fx smaller)",
                _mb(int8_path), _mb(fp32_path) / _mb(int8_path))

    # The tokenizer is pure Python/Rust and stays as-is.
    tok_dir = ONNX_DIR / "tokenizer"
    tokenizer.save_pretrained(str(tok_dir))
    logger.info("  tokenizer saved to %s", tok_dir.relative_to(PROJECT_ROOT))


def export_cnn() -> None:
    """Export the heatmap CNN to ONNX, then quantize to int8."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    fp32_path = ONNX_DIR / "cnn_fp32.onnx"
    int8_path = ONNX_DIR / "cnn_int8.onnx"

    logger.info("Exporting CNN to ONNX...")
    model = CNN()
    model.load_state_dict(torch.load(
        MODELS_DIR / "cnn_classifier.pt", map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        # Legacy TorchScript exporter. torch>=2.9 defaults to the dynamo path,
        # whose graphs fail onnxruntime's shape inference during quantization
        # ("Inferred shape and existing shape differ").
        dynamo=False,
    )
    logger.info("  fp32 ONNX: %.1f MB", _mb(fp32_path))

    logger.info("  quantizing to int8...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    logger.info("  int8 ONNX: %.1f MB  (%.1fx smaller)",
                _mb(int8_path), _mb(fp32_path) / _mb(int8_path))


def smoke_test() -> None:
    """Confirm the quantized graphs load and produce sane probabilities."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    logger.info("Smoke-testing quantized models...")

    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR / "tokenizer"))
    sess = ort.InferenceSession(str(ONNX_DIR / "transformer_int8.onnx"),
                                providers=["CPUExecutionProvider"])
    tokens = tokenizer("This is a short piece of text used to verify the graph runs.",
                       max_length=MAX_TOKENS, padding="max_length",
                       truncation=True, return_tensors="np")
    logits = sess.run(None, {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64),
    })[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    logger.info("  transformer P(AI) = %.4f", probs[0][1])

    cnn_sess = ort.InferenceSession(str(ONNX_DIR / "cnn_int8.onnx"),
                                    providers=["CPUExecutionProvider"])
    cnn_logits = cnn_sess.run(None, {
        "images": np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)})[0]
    cnn_probs = np.exp(cnn_logits) / np.exp(cnn_logits).sum(axis=1, keepdims=True)
    logger.info("  cnn P(AI) = %.4f", cnn_probs[0][1])
    logger.info("  both graphs run")


def main():
    parser = argparse.ArgumentParser(description="Export + quantize models to int8 ONNX")
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--keep-fp32", action="store_true",
                        help="Keep the intermediate fp32 ONNX files")
    args = parser.parse_args()

    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_transformer:
        export_transformer()
    if not args.skip_cnn:
        export_cnn()

    smoke_test()

    if not args.keep_fp32:
        for name in ("transformer_fp32.onnx", "cnn_fp32.onnx"):
            path = ONNX_DIR / name
            if path.exists():
                path.unlink()
        # Large models export external weight files alongside the graph.
        for extra in ONNX_DIR.glob("*.onnx.data"):
            extra.unlink()
        logger.info("Removed intermediate fp32 graphs (--keep-fp32 to retain)")

    total = sum(_mb(p) for p in ONNX_DIR.rglob("*") if p.is_file())
    logger.info("Deployment payload in %s: %.1f MB",
                ONNX_DIR.relative_to(PROJECT_ROOT), total)
    logger.info("Next: python src/evaluate_onnx.py --dataset raid")


if __name__ == "__main__":
    main()
