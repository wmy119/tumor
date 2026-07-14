import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.datasets.brats_seg_dataset import BraTSSegmentationDataset
from src.models.unet import UNet3D


def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_dir = cfg["data"].get("train_dir", cfg["data"]["preprocessed_dir"])
    val_dir = cfg["data"].get("val_dir")
    target_shape = cfg["preprocessing"]["target_shape"]

    if not val_dir:
        raise ValueError("请在 config/config.yaml 中配置 data.val_dir，用于验证集目录。")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练目录不存在: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"验证目录不存在: {val_dir}")

    train_ds = BraTSSegmentationDataset(train_dir, target_shape=target_shape)
    val_ds = BraTSSegmentationDataset(val_dir, target_shape=target_shape)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["segmentation"]["batch_size"],
        shuffle=True,
        num_workers=cfg["segmentation"].get("num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["segmentation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["segmentation"].get("num_workers", 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=4, num_classes=cfg["segmentation"]["num_classes"]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["segmentation"]["learning_rate"])

    for epoch in range(cfg["segmentation"]["epochs"]):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader) if len(train_loader) else 0.0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader) if len(val_loader) else 0.0

        print(
            f"Epoch {epoch+1}/{cfg['segmentation']['epochs']} - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
