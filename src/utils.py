import os
import numpy as np
import matplotlib.pyplot as plt


"""工具函数集：包含文件夹确保与可视化辅助函数。

函数:
- `ensure_dir(path)`: 确保目录存在（创建不存在的目录）
- `visualize_slice(image, label=None, save_path=None)`: 绘制单张切片并可选保存
"""


def ensure_dir(path):
    """确保传入的目录路径存在（会创建所有父目录）。"""
    os.makedirs(path, exist_ok=True)


def visualize_slice(image, label=None, save_path=None):
    """可视化单张灰度切片并叠加轮廓分割（若提供）。

    参数:
    - image: 2D numpy 数组的灰度图像
    - label: 可选的 2D 分割数组（将以轮廓形式叠加）
    - save_path: 可选的保存路径，若提供则会创建父目录并保存图片
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    if label is not None:
        # 以一条轮廓线绘制标签边界，便于观察分割位置
        ax.contour(label, levels=[1], colors="red", alpha=0.6)
    ax.axis("off")
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
