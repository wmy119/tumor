import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.preprocessing import preprocess_case


"""
分割数据集包装器，用于语义分割任务。

支持两种输入目录类型：
- 预处理后的 `.npz` 文件目录，每个样本包含 `images` 和 `label`
- 原始 BraTS 案例目录，每个样本目录包含 `T1.nii.gz`, `T1ce.nii.gz`,
  `T2.nii.gz`, `FLAIR.nii.gz` 和 `seg.nii.gz`

返回：
- `images`: torch.FloatTensor，形状 (C, Z, Y, X)
- `labels`: torch.LongTensor 或 IntTensor，形状 (Z, Y, X)
"""


class BraTSSegmentationDataset(Dataset):
    def __init__(self, data_dir, target_shape=None):
        """初始化分割数据集。

        此方法用于初始化一个分割数据集实例，支持从预处理的 .npz 文件或原始 BraTS 案例目录加载数据。
        参数:
        - data_dir: 包含 `.npz` 文件或原始 BraTS 案例目录的路径
        - target_shape: 仅在原始 BraTS 数据目录中需要，用于预处理尺寸
        """
        # 存储数据目录路径
        self.data_dir = data_dir
        # 存储目标形状（用于原始数据预处理）
        self.target_shape = target_shape
        # 初始化样本列表
        self.samples = []
        # 标记是否为原始数据（非 .npz 格式）
        self.raw = False

        entries = sorted(os.listdir(data_dir))
        npz_samples = [file[:-4] for file in entries if file.endswith(".npz")]
        if npz_samples:
            self.samples = npz_samples
        else:
            case_dirs = [entry for entry in entries if os.path.isdir(os.path.join(data_dir, entry))]
            self.samples = case_dirs
            self.raw = len(case_dirs) > 0

        # 如果是原始数据且没有提供目标形状，抛出异常
        if self.raw and self.target_shape is None:
            raise ValueError("Raw BraTS 数据目录需要提供 target_shape 参数进行预处理。")

        # 如果没有找到有效样本，抛出异常
        if not self.samples:
            raise ValueError(f"未在目录中找到有效样本: {data_dir}")

    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.samples)

    def __getitem__(self, idx):
        """返回单个样例：图像张量与标签张量。"""
        sample_name = self.samples[idx]

        if self.raw:
            case_dir = os.path.join(self.data_dir, sample_name)
            images, label = preprocess_case(case_dir, self.target_shape)
        else:
            data_path = os.path.join(self.data_dir, f"{sample_name}.npz")
            data = np.load(data_path)
            images = data["images"].astype(np.float32)
            label = data["label"].astype(np.int64)

        images = torch.from_numpy(images)
        labels = torch.from_numpy(label)
        return images, labels
