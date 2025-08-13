#!/usr/bin/env python
# coding: utf-8
import os
import sys
import math
from collections import deque
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 配置（与导出脚本、model.json 保持一致）
# -------------------------------
DATASET_NAME = "TINY_IMAGENET"
BATCH_SIZE = 128
NUM_CLASSES = 200
IMG_H = 64
IMG_W = 64
IN_CH = 3
IMG_ELEM = IMG_H * IMG_W * IN_CH  # 12288
LR = 2 ** -3  # 0.125

# Tiny ImageNet（导出脚本使用 drop_last=True）有效样本/批次数
TRAIN_EFFECTIVE_SAMPLES = 99968
VAL_EFFECTIVE_SAMPLES = 9984
TRAIN_NUM_BATCHES = TRAIN_EFFECTIVE_SAMPLES // BATCH_SIZE  # 781
VAL_NUM_BATCHES = VAL_EFFECTIVE_SAMPLES // BATCH_SIZE      # 78

# 路径：假设本文件与导出脚本同级
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(THIS_DIR, "train_data")
TRAIN_LABELS_PATH = os.path.join(THIS_DIR, "train_labels")
TEST_DATA_PATH = os.path.join(THIS_DIR, "test_data")
TEST_LABELS_PATH = os.path.join(THIS_DIR, "test_labels")


# -------------------------------
# 数据读取：按 batch 流式从文本中读取
# -------------------------------
def read_n_floats_text(fp, n):
    """
    从文本文件句柄中读取 n 个浮点数（逐行一个值），返回 float32 的一维 ndarray。
    如果提前 EOF，会返回数量不足的数组。
    """
    # 读取 n 行文本
    lines = list(islice(fp, n))
    if not lines:
        return np.array([], dtype=np.float32)
    # 使用 numpy 向量化解析为 float32
    arr = np.array(lines, dtype=np.float32)
    if arr.size != n:
        # 提醒读取数量不足（通常意味着文件结束或文件不完整）
        sys.stderr.write(f"Warning: expected {n} floats, got {arr.size}\n")
    return arr


def batch_generator_from_txt(data_path, labels_path, batch_size, num_batches,
                             img_h=64, img_w=64, in_ch=3, num_classes=200):
    """
    逐 batch 从导出文本读取，并产出 (x, y_idx)：
      - x: float32, [B, C, H, W]
      - y_idx: long, [B]（由 one-hot 取 argmax 得到的类别索引）
    注意：导出时保存排列为 [B, W, H, C] 且逐值换行，这里要转回 [B, C, H, W]。
    """
    elems_per_img = img_h * img_w * in_ch
    elems_per_label = num_classes

    with open(data_path, "r") as f_img, open(labels_path, "r") as f_lbl:
        for b in range(num_batches):
            # 读取一个 batch 的像素与标签
            img_vals = read_n_floats_text(f_img, batch_size * elems_per_img)
            lbl_vals = read_n_floats_text(f_lbl, batch_size * elems_per_label)

            if img_vals.size < batch_size * elems_per_img or lbl_vals.size < batch_size * elems_per_label:
                # 文件到头，提前结束
                break

            # 还原形状：导出为 [B, W, H, C]，转回 [B, C, H, W]
            imgs_bwhc = img_vals.reshape(batch_size, img_w, img_h, in_ch)
            imgs = np.transpose(imgs_bwhc, (0, 3, 2, 1)).astype(np.float32)  # [B, C, H, W]

            labels_oh = lbl_vals.reshape(batch_size, num_classes).astype(np.float32)
            labels_idx = np.argmax(labels_oh, axis=1).astype(np.int64)

            # 转为 torch tensor（CPU）
            x = torch.from_numpy(imgs)             # float32
            y = torch.from_numpy(labels_idx)       # int64

            yield x, y


# -------------------------------
# 模型定义：完全按给定 JSON 拟合
# -------------------------------
class AlexNetTIN64(nn.Module):
    """
    严格按 model.json 中的层次与超参定义：
    1) Conv(3->96, k=11, s=4, p=1) -> AvgPool(3, s=3) -> ReLU
    2) Conv(96->256, k=5, s=1, p=1) -> AvgPool(2, s=2) -> ReLU
    3) Conv(256->384, k=3, s=1, p=1) -> ReLU
    4) Conv(384->384, k=3, s=1, p=1) -> ReLU
    5) Conv(384->256, k=3, s=1, p=1) -> ReLU
    6) FC(256->256) -> ReLU
    7) FC(256->256) -> ReLU
       FC(256->200)
    """
    def __init__(self, num_classes=200):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=1, bias=True),  # -> [96,14,14]
            nn.AvgPool2d(kernel_size=3, stride=3),  # -> [96,4,4]
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=1, bias=True),  # -> [256,2,2]
            nn.AvgPool2d(kernel_size=2, stride=2),  # -> [256,1,1]
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),  # -> [384,1,1]
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),  # -> [384,1,1]
            nn.ReLU(inplace=True),

            # Block 5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),  # -> [256,1,1]
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 256*1*1 -> 256
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------------
# 训练与评估
# -------------------------------
def train_one_epoch(model, optimizer, loss_fn, epoch, num_batches):
    model.train()
    loss_window = deque(maxlen=20)

    gen = batch_generator_from_txt(
        TRAIN_DATA_PATH, TRAIN_LABELS_PATH,
        batch_size=BATCH_SIZE, num_batches=num_batches,
        img_h=IMG_H, img_w=IMG_W, in_ch=IN_CH, num_classes=NUM_CLASSES
    )

    for b_idx, (x, y) in enumerate(gen, start=1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # [B, 200]
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        loss_window.append(loss.item())
        # 输出大小为20的滑动窗口 loss（不满20时取当前已有均值）
        avg20 = sum(loss_window) / len(loss_window)
        print(f"Epoch {epoch} | Batch {b_idx}/{num_batches} | loss(win20)={avg20:.6f}", flush=True)


@torch.no_grad()
def evaluate_top1(model, num_batches):
    model.eval()
    total = 0
    correct = 0

    gen = batch_generator_from_txt(
        TEST_DATA_PATH, TEST_LABELS_PATH,
        batch_size=BATCH_SIZE, num_batches=num_batches,
        img_h=IMG_H, img_w=IMG_W, in_ch=IN_CH, num_classes=NUM_CLASSES
    )

    for x, y in gen:
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    acc = correct / total if total > 0 else 0.0
    print(f"Validation top-1 accuracy: {acc*100:.2f}%")
    return acc


def main():
    # 基本检查
    for p in [TRAIN_DATA_PATH, TRAIN_LABELS_PATH, TEST_DATA_PATH, TEST_LABELS_PATH]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing file: {p}")

    # 仅使用 CPU
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = torch.device("cpu")

    # 构建模型（严格对齐 JSON）
    model = AlexNetTIN64(num_classes=NUM_CLASSES).to(device)

    # 优化器：SGD 固定学习率 2^-3
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # 损失函数：CrossEntropy（labels 为 one-hot，取 argmax 后作为类别索引）
    loss_fn = nn.CrossEntropyLoss()

    # 训练轮数（可根据需要调整）
    epochs = 1
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, optimizer, loss_fn, epoch, num_batches=TRAIN_NUM_BATCHES)
        evaluate_top1(model, num_batches=VAL_NUM_BATCHES)


if __name__ == "__main__":
    main()
