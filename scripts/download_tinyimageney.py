#!/usr/bin/env python
# coding: utf-8

import os
import random
import shutil
import zipfile
import urllib.request
import sys

import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Config: keep this name in sync with model.json's "dataset"
DATASET_NAME = "TINY_IMAGENET"
OUTPUT_DIR = f"../files/{DATASET_NAME}/"
BATCH_SIZE = 128  # keep in sync with model.json's "batch_size"
NUM_CLASSES = 200
OUTPUT_FMT = "%.6f"  # 输出精度

def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    oh = np.zeros((labels.size, num_classes), dtype=np.float32)
    oh[np.arange(labels.size), labels] = 1.0
    return oh

def append_array_to_txt(arr: np.ndarray, filename: str, fmt: str = OUTPUT_FMT):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "ab") as f:
        # 写为纯文本，空白分隔；使用指定精度
        np.savetxt(f, arr.reshape(-1), fmt=fmt)

def reset_file(path: str):
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

def download_and_prepare_tiny_imagenet(target_dir: str = "./") -> str:
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(target_dir, "tiny-imagenet-200.zip")
    root = os.path.join(target_dir, "tiny-imagenet-200")

    if not os.path.exists(root):
        os.makedirs(target_dir, exist_ok=True)
        if not os.path.exists(zip_path):
            print("Downloading Tiny ImageNet (~237MB)...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete:", zip_path)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        print("Extracted to:", root)

        # Reorganize val for ImageFolder
        val_dir = os.path.join(root, "val")
        val_images_dir = os.path.join(val_dir, "images")
        ann_path = os.path.join(val_dir, "val_annotations.txt")

        mapping = {}
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img, wnid = parts[0], parts[1]
                    mapping[img] = wnid

        print("Reorganizing val images into class subfolders...")
        for img, wnid in mapping.items():
            src = os.path.join(val_images_dir, img)
            if not os.path.exists(src):
                continue
            dst_dir = os.path.join(val_dir, wnid)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, img)
            shutil.move(src, dst)

        try:
            shutil.rmtree(val_images_dir)
        except Exception:
            pass

        print("Val reorganization complete.")

    return root

def print_progress(iter_idx: int, total: int, prefix: str):
    bar_len = 30
    filled = int(bar_len * iter_idx / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = 100.0 * iter_idx / total
    msg = f"\r{prefix} [{bar}] {iter_idx}/{total} ({pct:5.1f}%)"
    print(msg, end="", flush=True)

if __name__ == "__main__":
    TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, "train_data")
    TRAIN_LABELS_PATH = os.path.join(OUTPUT_DIR, "train_labels")
    TEST_DATA_PATH = os.path.join(OUTPUT_DIR, "test_data")
    TEST_LABELS_PATH = os.path.join(OUTPUT_DIR, "test_labels")

    dataset_root = download_and_prepare_tiny_imagenet("./")

    transform = transforms.ToTensor()  # [0,1], CHW float32
    train_dataset = datasets.ImageFolder(os.path.join(dataset_root, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_root, "val"), transform=transform)  # as "test"

    assert len(train_dataset.classes) == NUM_CLASSES, "NUM_CLASSES mismatch with dataset folders"

    # 仅对训练集进行打乱；使用固定种子的 generator 确保可复现
    gen = torch.Generator()
    gen.manual_seed(0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,           # 改为打乱
        drop_last=True,
        num_workers=2,
        generator=gen           # 固定随机种子
    )
    # 验证集保持不打乱
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=2)

    print(f"Classes: {len(train_dataset.classes)}")
    print(f"Train images (effective): {len(train_loader) * BATCH_SIZE}")
    print(f"Val images (effective):   {len(val_loader) * BATCH_SIZE}")
    print(f"Output precision: {OUTPUT_FMT}")

    # Initialize output files
    reset_file(TRAIN_DATA_PATH)
    reset_file(TRAIN_LABELS_PATH)
    reset_file(TEST_DATA_PATH)
    reset_file(TEST_LABELS_PATH)

    # Stream out train
    print("Writing train data/labels...")
    total_train_batches = len(train_loader)
    for b_idx, (imgs, labels) in enumerate(train_loader, start=1):
        # imgs: [B, C, H, W] -> [B, W, H, C] to match existing MNIST/CIFAR exporters
        arr = imgs.numpy().transpose(0, 3, 2, 1).astype(np.float32)
        append_array_to_txt(arr, TRAIN_DATA_PATH, fmt=OUTPUT_FMT)
        append_array_to_txt(one_hot(labels.numpy(), NUM_CLASSES), TRAIN_LABELS_PATH, fmt=OUTPUT_FMT)

        if b_idx == 1 or b_idx % 10 == 0 or b_idx == total_train_batches:
            print_progress(b_idx, total_train_batches, prefix="Train")

    print()  # newline after progress bar

    # Stream out val (as test)
    print("Writing val (as test) data/labels...")
    total_val_batches = len(val_loader)
    for b_idx, (imgs, labels) in enumerate(val_loader, start=1):
        arr = imgs.numpy().transpose(0, 3, 2, 1).astype(np.float32)
        append_array_to_txt(arr, TEST_DATA_PATH, fmt=OUTPUT_FMT)
        append_array_to_txt(one_hot(labels.numpy(), NUM_CLASSES), TEST_LABELS_PATH, fmt=OUTPUT_FMT)

        if b_idx == 1 or b_idx % 10 == 0 or b_idx == total_val_batches:
            print_progress(b_idx, total_val_batches, prefix="Val  ")

    print()  # newline after progress bar

    print("Done. Saved to:", OUTPUT_DIR)
    print("Note: Tiny ImageNet official test has no labels; we use 'val' as 'test'.")
