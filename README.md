# BraTS2020 Glioma Segmentation and Classification Project

This project provides a starter framework for a glioma MRI analysis pipeline using BraTS2020 data.

## Pipeline Overview

1. Data preprocessing for 4 MRI modalities: T1, T1ce, T2, FLAIR
2. Tumor segmentation using a U-Net based model
3. Tumor-level classification using a ResNet based model
4. Visualization and evaluation utilities

## Recommended Directory Structure

- data/: raw and preprocessed datasets
- config/: configuration files
- src/: source code
  - preprocessing.py
  - datasets/
  - models/
  - utils/
  - train_seg.py
  - train_cls.py

## Quick Start

```bash
pip install -r requirements.txt
python src/train_seg.py --config config/config.yaml
python src/train_cls.py --config config/config.yaml
```

## Notes

- BraTS2020 data should be placed under data/raw
- Labels are expected in NIfTI (.nii.gz) format
- The current code is a structured starter scaffold and can be extended for full training
