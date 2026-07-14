import argparse
import os
import random
from pathlib import Path


def get_sample_names(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")

    samples = []
    for entry in sorted(data_dir.iterdir()):
        if entry.is_dir():
            samples.append(entry.name)
        elif entry.is_file() and entry.suffix == ".npz":
            samples.append(entry.stem)

    if not samples:
        raise ValueError(f"未在目录中找到样本：{data_dir}")

    return samples


def write_split_file(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")


def main():
    parser = argparse.ArgumentParser(description="生成 BraTS split 文件（train/val/test）。")
    parser.add_argument("--source-dir", required=True, help="包含样本目录或 .npz 文件的源目录。")
    parser.add_argument("--output-dir", default="data/splits", help="生成 split 文件的目录。默认 data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例，默认 0.8")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例，默认 0.1")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子，保证 split 可复现")
    args = parser.parse_args()

    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio 和 val_ratio 必须在 0~1 之间，且 train_ratio + val_ratio < 1")

    names = get_sample_names(args.source_dir)
    random.Random(args.seed).shuffle(names)

    n = len(names)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_names = names[:n_train]
    val_names = names[n_train:n_train + n_val]
    test_names = names[n_train + n_val:]

    out_dir = Path(args.output_dir)
    write_split_file(out_dir / "train.txt", train_names)
    write_split_file(out_dir / "val.txt", val_names)
    write_split_file(out_dir / "test.txt", test_names)

    print(f"生成完成: {len(train_names)} train, {len(val_names)} val, {len(test_names)} test")
    print(f"文件位置: {out_dir / 'train.txt'}, {out_dir / 'val.txt'}, {out_dir / 'test.txt'}")


if __name__ == "__main__":
    main()
