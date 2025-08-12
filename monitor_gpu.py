import csv
import subprocess
import time
from datetime import datetime
import signal
import argparse

interval = 0.25
filename = "gpu_utilization_log.csv"

def get_gpu_utilization():
    """
    获取第一块NVIDIA显卡的GPU利用率。
    """
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            "--id=0"
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        utilization = int(result.stdout.strip())
        return utilization
    except FileNotFoundError:
        print("错误: `nvidia-smi` 命令未找到。请确保已正确安装NVIDIA驱动程序。")
        return None
    except Exception as e:
        print(f"获取GPU利用率时发生错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="记录 GPU 利用率到 CSV，可设置最长记录时长。")
    parser.add_argument(
        "-t", "--max-duration",
        type=float,
        default=None,
        help="最长记录时长（秒）。省略则无限制。"
    )
    args = parser.parse_args()

    interval = 0.25
    max_duration = args.max_duration
    data_points = []

    # 标记是否已遇到第一个非0利用率，以及其对应的起始时间（单调时钟）
    started = False
    t0_monotonic = None

    def save_csv():
        if not data_points:
            return
        import csv
        with open("gpu_utilization_log.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # 相对时间（秒），相对于第一次非0利用率的时刻
            w.writerow(["Relative_Time_s", "GPU_Utilization_%"])
            w.writerows(data_points)

    def handle_signal(signum, frame):
        # 统一把 SIGINT/SIGTERM 转成 KeyboardInterrupt，落到 finally 保存
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    start_monotonic = time.monotonic()
    deadline = start_monotonic + max_duration if max_duration is not None else None

    try:
        while True:
            # 若设置了最长时长且已到达/超过，立即退出
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print("达到最长记录时长，准备退出并保存数据...")
                    break
            else:
                remaining = None

            utilization = get_gpu_utilization()
            t_now = time.monotonic()  # 以单调时钟作为时间基准，避免系统时间跳变

            if utilization is not None:
                if not started:
                    # 只有在首次出现非0利用率时开始记录，且时间为0.0
                    if utilization > 0:
                        started = True
                        t0_monotonic = t_now
                        data_points.append([0.0, utilization])
                else:
                    # 已开始后，持续记录（包括后续的0值），时间为相对t0的秒数
                    rel_t = t_now - t0_monotonic
                    # 以毫秒精度保留三位小数
                    data_points.append([round(rel_t, 3), utilization])

            # 为了在接近截止时间时更快退出，睡眠不超过剩余时间
            if remaining is None:
                time.sleep(interval)
            else:
                sleep_time = max(0.0, min(interval, remaining))
                # 如果已到截止瞬间，直接继续循环让上方检查触发退出
                if sleep_time == 0.0:
                    continue
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("收到信号，准备退出并保存数据...")
    finally:
        save_csv()
        print("已保存 CSV。")

if __name__ == "__main__":
    main()
