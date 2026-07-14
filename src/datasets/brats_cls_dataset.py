import os
import numpy as np
import torch
from torch.utils.data import Dataset


"""
分类数据集包装器（用于二分类：有无肿瘤）。

数据源期望为预处理后的 `.npz` 文件，每个文件包含两个键：
- `images`: 形状 (C, Z, Y, X) 的多模态图像
- `label`: 形状 (Z, Y, X) 的标签体积

本类将 `label` 简化为二值：只要体积分割中存在任意非零像素，则视为阳性样例（label=1），否则为阴性（label=0）。
"""


class BraTSClassificationDataset(Dataset):
    def __init__(self, data_dir):
        """初始化数据集。

        参数:
        - data_dir: 包含 `.npz` 文件的目录
        """
        self.data_dir = data_dir
        self.samples = []

        for file in sorted(os.listdir(data_dir)):
            if file.endswith(".npz"):
                self.samples.append(file[:-4])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回单个样例：images 张量与二值标签。

        - images: torch.FloatTensor, 形状 (C, Z, Y, X)
        - label: torch.LongTensor, 标量（0 或 1）
        """
        sample_name = self.samples[idx]
        data_path = os.path.join(self.data_dir, f"{sample_name}.npz")
        data = np.load(data_path)

        # 读取图像数据并保证 dtype
        images = data["images"].astype(np.float32)
        # 将分割体积转换为二分类标签：只要有任意非零像素则视为阳性
        label = int(np.max(data["label"]) > 0)

        images = torch.from_numpy(images)
        return images, torch.tensor(label, dtype=torch.long)
