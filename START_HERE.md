# Start Here

This project is now scaffolded for a BraTS2020-based glioma MRI segmentation and classification workflow.

## What has been created

- Project documentation: README.md
- Python dependencies: requirements.txt
- Configuration file: config/config.yaml
- Preprocessing pipeline: src/preprocessing.py
- Segmentation dataset: src/datasets/brats_seg_dataset.py
- Classification dataset: src/datasets/brats_cls_dataset.py
- U-Net model: src/models/unet.py
- ResNet-based classifier: src/models/resnet.py
- Training scripts: src/train_seg.py and src/train_cls.py

## Next steps

1. Put your BraTS2020 data under data/raw
2. Create split files under data/splits
3. Run preprocessing:

```bash
python -c "from src.preprocessing import preprocess_dataset; preprocess_dataset('data/raw', 'data/preprocessed', [128,128,128])"
```

4. Start segmentation training:

```bash
python src/train_seg.py
```

5. Start classification training:

```bash
python src/train_cls.py
```

## Notes

- The current code is a starter framework and should be expanded with proper train/val/test splitting, metric calculation, and 3D patch-based training for better performance.
