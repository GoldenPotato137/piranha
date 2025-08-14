#!/usr/bin/env python3
import numpy as np
from pathlib import Path

# ===== 配置，根据你的实际情况调整 =====
DATASET_DIR = Path(".")  # 改成你的数据目录
TRAIN_DATA = DATASET_DIR / "train_data"
BATCH_SIZE = 128
C, H, W = 3, 64, 64  # TinyImageNet + AlexNet
INPUT_SIZE = C * H * W
NUM_BATCHES_TO_PRINT = 3  # 打印前 3 个 batch

def compute_stats(arr: np.ndarray):
    # arr 为 1D 向量
    mn = float(arr.min())
    mx = float(arr.max())
    mean = float(arr.mean())
    std = float(arr.std())
    return mn, mx, mean, std

def compute_channel_stats_NCHW(batch_flat: np.ndarray, B: int, C: int, H: int, W: int):
    # 假设扁平向量顺序为 NCHW，返回每通道统计
    x = batch_flat.reshape(B, C, H, W)
    stats = []
    for c in range(C):
        ch = x[:, c, :, :].reshape(-1)
        stats.append(compute_stats(ch))
    return stats

def compute_channel_stats_NHWC(batch_flat: np.ndarray, B: int, C: int, H: int, W: int):
    # 假设扁平向量顺序为 NHWC，返回每通道统计
    x = batch_flat.reshape(B, H, W, C)
    stats = []
    for c in range(C):
        ch = x[:, :, :, c].reshape(-1)
        stats.append(compute_stats(ch))
    return stats

def iter_batches_space_separated(path: Path, batch_size: int, input_size: int, dtype=np.float64, sep: str = ' '):
    """
    以空白分隔的 ASCII 浮点文本，按需流式读取。
    每次最多读取 batch_size 个样本（batch_size * input_size 个标量）。
    如遇文件尾：
      - 若能组成若干完整样本（个数是 input_size 的整数倍），则返回较小的最后一个 batch；
      - 若末尾出现残缺样本（不是 input_size 的整数倍），则丢弃尾部多余标量并告警。
    产出：(batch_flat, B_actual)
      - batch_flat: 1D ndarray，长度 = B_actual * input_size
      - B_actual: 实际样本数（<= batch_size）
    """
    need_per_batch = batch_size * input_size
    with open(path, 'r') as f:
        while True:
            vals = np.fromfile(f, sep=sep, dtype=dtype, count=need_per_batch)
            if vals.size == 0:
                break

            if vals.size < need_per_batch:
                # 文件已经接近结尾
                remainder = vals.size % input_size
                if remainder != 0:
                    # 丢弃无法组成完整样本的尾部
                    keep = vals.size - remainder
                    if keep == 0:
                        print("[WARN] Encountered trailing incomplete sample; no full sample remains. Stopping.")
                        break
                    print(f"[WARN] Dropping {remainder} trailing scalars from an incomplete sample at EOF.")
                    vals = vals[:keep]

            B_actual = vals.size // input_size
            if B_actual == 0:
                break

            # 如果恰好读满 need_per_batch，B_actual == batch_size；否则是最后一个较小的 batch
            yield vals, B_actual

def main():
    if not TRAIN_DATA.is_file():
        raise FileNotFoundError(f"Missing file: {TRAIN_DATA}")

    print(f"Streaming from {TRAIN_DATA} ...")

    batches_printed = 0
    for b, (batch, B_actual) in enumerate(iter_batches_space_separated(TRAIN_DATA, BATCH_SIZE, INPUT_SIZE, dtype=np.float64, sep=' ')):
        if batches_printed >= NUM_BATCHES_TO_PRINT:
            break

        if B_actual < BATCH_SIZE:
            print(f"[INFO] Last/short batch {b}: using B={B_actual} (< {BATCH_SIZE})")

        # 计算统计量
        g_mn, g_mx, g_mean, g_std = compute_stats(batch)
        nchw_stats = compute_channel_stats_NCHW(batch, B_actual, C, H, W)
        nhwc_stats = compute_channel_stats_NHWC(batch, B_actual, C, H, W)

        print(f"[INPUT] tag=train batch={b} B={B_actual} C={C} H={H} W={W}")
        print(f"  Global: min={g_mn:.6f} max={g_mx:.6f} mean={g_mean:.6f} std={g_std:.6f}")
        print("  Per-channel (assuming NCHW):")
        for c, (mn, mx, mean, std) in enumerate(nchw_stats):
            print(f"    C{c}: min={mn:.6f} max={mx:.6f} mean={mean:.6f} std={std:.6f}")
        print("  Per-channel (assuming NHWC):")
        for c, (mn, mx, mean, std) in enumerate(nhwc_stats):
            print(f"    C{c}: min={mn:.6f} max={mx:.6f} mean={mean:.6f} std={std:.6f}")

        # 保存首个样本前 64 个扁平元素，便于与 C++ 的 debug_input_*_head.txt 对比
        head = min(64, INPUT_SIZE)
        head_vals = batch[:head]
        out_path = DATASET_DIR / f"py_debug_input_train_batch{b}_first_sample_head.txt"
        np.savetxt(out_path, head_vals[None, :], fmt="%.10f", delimiter=" ")
        print(f"  Wrote head of first sample to: {out_path}")

        # 如需保存整个首样本的扁平向量，可解除下行注释（将写入 12288 个数）
        # full_path = DATASET_DIR / f"py_debug_input_train_batch{b}_first_sample_full.txt"
        # np.savetxt(full_path, batch[:INPUT_SIZE], fmt="%.10f")
        # print(f"  Wrote full first sample to: {full_path}")

        batches_printed += 1

    if batches_printed == 0:
        print("[WARN] No batch was produced from the file. Check INPUT_SIZE and file format.")

if __name__ == "__main__":
    main()
