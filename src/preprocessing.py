import os
import numpy as np
import nibabel as nib
from pathlib import Path


"""
预处理模块

本文件包含用于处理 BraTS 数据集的常用辅助函数：
- 读取 NIfTI 文件并返回数组与仿射矩阵
- 对体积进行非零像素归一化
- 对体积进行补零或裁剪以匹配目标尺寸
- 针对单个样例与整个数据集的预处理流水线

主要目的是把来自原始 NIfTI 的多模态影像处理成 numpy 数组并保存为压缩的 `.npz` 文件，便于训练时快速加载。
"""


def load_nifti(path):
    """读取 NIfTI 文件并返回数据数组与仿射矩阵。

    参数:
    - path: NIfTI 文件路径（例如 `.nii` 或 `.nii.gz`）

    返回:
    - volume: numpy 数组（dtype=np.float32）
    - affine: 仿射矩阵（用于坐标变换，当前函数只返回并不使用）
    """
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img.affine


def normalize_volume(volume):
    """对体积图像执行基于非零像素的 z-score 归一化。

    说明:
    - 只对非零像素计算均值与标准差，避免背景值（通常为 0）干扰统计量。
    - 若没有非零像素或标准差为 0，则返回原图以避免除以 0 的情况。
    """
    nonzero = volume[np.nonzero(volume)]
    if len(nonzero) == 0:
        return volume
    mean = nonzero.mean()
    std = nonzero.std()
    if std == 0:
        return volume
    return (volume - mean) / std


def pad_or_crop(volume, target_shape):
    """将三维体积通过补零或裁剪以匹配目标尺寸。

    - 输入 `volume` 期望为 (Z, Y, X) 形状。
    - 如果体积尺寸大于目标尺寸则在每个维度上裁剪到最小值；
      如果小于目标则用 0 补齐到目标尺寸（在前端对齐）。

    注意: 当前实现简单地在起始端对齐（即取前面的切片/放在数组前部），
    若需中心对齐或基于配准的裁剪，请修改此函数。
    """
    # 将输入转换为NumPy数组，确保我们可以使用NumPy的操作
    volume = np.asarray(volume)
    # 如果当前体积尺寸已经与目标尺寸匹配，直接返回
    if volume.shape == tuple(target_shape):
        return volume

    # 解包目标形状和当前形状的各个维度
    target_z, target_y, target_x = target_shape
    current_z, current_y, current_x = volume.shape

    # 创建一个与目标形状相同，但全部填充0的新数组
    new_volume = np.zeros(target_shape, dtype=np.float32)

    # 取每个维度的最小重叠区域并复制过去
    new_z = min(target_z, current_z)
    new_y = min(target_y, current_y)
    new_x = min(target_x, current_x)

    # 将原始体积数据复制到新体积数组的相应区域
    # 使用切片操作确保只复制有效区域，避免越界
    new_volume[:new_z, :new_y, :new_x] = volume[:new_z, :new_y, :new_x]
    # 返回处理后的新体积数组
    return new_volume


def preprocess_case(case_dir, target_shape):
    """对单个病人(case)目录执行预处理。

    期望目录结构（每个 case）:
    - T1.nii.gz, T1ce.nii.gz, T2.nii.gz, FLAIR.nii.gz, seg.nii.gz

    返回:
    - images: numpy 数组，形状为 (C, Z, Y, X)，C=4（4 种模态）
    - label: 整数标签体积，形状为 (Z, Y, X)

    步骤:
    1. 依次读取每种模态的 NIfTI 文件
    2. 对每个模态执行基于非零像素的归一化
    3. 对图像与标签执行补零/裁剪以匹配 `target_shape`
    4. 将模态维度合并为通道维并返回
    """
    modalities = ["T1", "T1ce", "T2", "FLAIR"]
    data = []

    for mod in modalities:
        # 构造每种模态的文件路径并加载
        path = os.path.join(case_dir, f"{mod}.nii.gz")
        vol, _ = load_nifti(path)
        # 按非零像素归一化以减少背景影响
        vol = normalize_volume(vol)
        # 调整体积大小以便后续批处理（相同形状）
        vol = pad_or_crop(vol, target_shape)
        data.append(vol)

    # 加载分割标签，四舍五入并转换为整数类型
    label_path = os.path.join(case_dir, "seg.nii.gz")
    label, _ = load_nifti(label_path)
    label = np.round(label).astype(np.int64)
    label = pad_or_crop(label, target_shape)

    # 返回 (C, Z, Y, X) 的图像数组以及 (Z, Y, X) 的标签
    return np.stack(data, axis=0), label


def preprocess_dataset(raw_dir, output_dir, target_shape):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    for case_dir in case_dirs:
        case_name = case_dir.name
        images, label = preprocess_case(case_dir, target_shape)
        np.savez_compressed(output_dir / f"{case_name}.npz", images=images, label=label)
        print(f"Processed {case_name}")
