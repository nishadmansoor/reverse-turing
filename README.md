# Reverse Turing Test 

An evaluation framework that compares how different classification paradigms — semantic (BERT), visual (OpenCV stylometric heatmaps), and human judgment — perform at distinguishing AI-generated text from human-written text.

## Components

| Module | Description |
|--------|-------------|
| `src/data_pipeline.py` | Load HC3, preprocess, train/val/test splits |
| `src/bert_classifier.py` | BERT-based semantic text classification (in progress) |
| `src/visual_features.py` | Convert text to stylometric heatmap images (OpenCV) (in progress) |
| `src/visual_classifier.py` | Image-based AI/human classification (in progress) |
| `src/evaluation.py` | Unified metrics & three-way comparison (in progress) |

