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
    def __init__(self, data_dir, split_file=None, target_shape=None):
        """初始化分割数据集。

        此方法用于初始化一个分割数据集实例，支持从预处理的 .npz 文件或原始 BraTS 案例目录加载数据。
        参数:
        - data_dir: 包含 `.npz` 文件或原始 BraTS 案例目录的路径
        - split_file: 可选的文本文件（每行一个样例名）用于指定子集
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

        # 如果提供了分割文件且文件存在
        if split_file and os.path.exists(split_file):
            # 读取分割文件内容
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    # 处理每行内容，去除前后空格
                    sample_name = line.strip()
                    if not sample_name:
                        continue
                    # 移除文件扩展名（如果存在）
                    if sample_name.endswith(".npz"):
                        sample_name = sample_name[:-4]
                    if sample_name.endswith(".nii.gz"):
                        sample_name = sample_name[:-7]
                    # 添加到样本列表
                    self.samples.append(sample_name)
        else:
            # 如果没有提供分割文件，则扫描数据目录
            entries = sorted(os.listdir(data_dir))
            # 查找所有 .npz 文件并移除扩展名
            npz_samples = [file[:-4] for file in entries if file.endswith(".npz")]
            if npz_samples:
                # 如果找到 .npz 文件，使用它们作为样本
                self.samples = npz_samples
            else:
                # 如果没有找到 .npz 文件，查找目录作为样本
                case_dirs = [entry for entry in entries if os.path.isdir(os.path.join(data_dir, entry))]
                self.samples = case_dirs
                # 标记为原始数据
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
