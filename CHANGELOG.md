# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Ideas
- Optional sentence-embedding text features (kept off by default, disk and CPU friendly first).
- Try CLIP image features as an optional extra.
- Docker file for the demo.

## [0.1.0] - 2026-09-30

### Added
- Notebook 01: download MVSA-Single, build labels, clean text, stratified train/val/test split, quick EDA.
- Notebook 02: frozen MobileNetV3-small image features, read directly from the zip file.
- Notebook 03: TF-IDF text features, text-only / image-only / early fusion / late fusion models, comparison
  table, confusion matrix, and saving the best model.
- Flask app with text + image input, per-modality and fused probabilities, example tweets, and the model comparison.
- README, CONTRIBUTE and CHANGELOG.
