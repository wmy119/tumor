import yaml
import torch
from torch.utils.data import DataLoader
from src.datasets.brats_cls_dataset import BraTSClassificationDataset
from src.models.resnet import ResNet3DClassifier


def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data"]["preprocessed_dir"]

    train_ds = BraTSClassificationDataset(data_dir)
    train_loader = DataLoader(train_ds, batch_size=cfg["classification"]["batch_size"], shuffle=True)

    model = ResNet3DClassifier(in_channels=4, num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["classification"]["learning_rate"])

    model.train()
    for epoch in range(cfg["classification"]["epochs"]):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{cfg['classification']['epochs']} - Loss: {running_loss/len(train_loader):.4f}")


if __name__ == "__main__":
    main()
