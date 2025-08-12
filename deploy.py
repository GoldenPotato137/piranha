#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import socket
import subprocess
import sys
import argparse
import stat
from typing import List, Dict, Optional

def get_local_ip() -> Optional[str]:
    """
    获取本机的局域网IP地址。
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except socket.error as e:
        print(f"错误: 无法获取本机IP地址: {e}")
        print("请检查您的网络连接。")
        return None

def main():
    parser = argparse.ArgumentParser(description="分发 piranha、配置文件和数据文件到远程参与方，并生成运行脚本。")
    parser.add_argument("config_path", help="要分发的配置文件的路径。")
    parser.add_argument("--send-files", action="store_true", help="如果指定，则同步位于脚本同级目录下的 'files' 目录。")
    args = parser.parse_args()

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
        
    config_file_path = os.path.abspath(args.config_path)
    
    if not os.path.isfile(config_file_path):
        print(f"错误: 提供的配置文件路径无效或不是一个文件: {config_file_path}")
        sys.exit(1)

    config_dir = os.path.dirname(config_file_path)
    piranha_file_path = os.path.join(script_dir, 'piranha')
    
    # 'files' 目录路径现在正确地指向脚本同级目录
    files_dir_path = os.path.join(script_dir, 'files')

    if not os.path.exists(piranha_file_path):
        print(f"错误: 在本机上未找到 'piranha' 二进制文件，期望路径为: {piranha_file_path}")
        sys.exit(1)

    files_dir_exists = os.path.isdir(files_dir_path)
    if args.send_files:
        if files_dir_exists:
            print(f"找到 'files' 目录，将进行同步: {files_dir_path}")
        else:
            print(f"警告: 指定了 --send-files，但在脚本目录 {script_dir} 中未找到 'files' 目录，将跳过同步。")
    else:
        print("未指定 --send-files，将跳过同步 'files' 目录。")

    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        party_ips: List[str] = config['party_ips']
        party_users: List[str] = config['party_users']
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"错误: 无法读取或解析配置文件 {config_file_path}: {e}")
        sys.exit(1)

    local_ip = get_local_ip()
    if not local_ip:
        sys.exit(1)
    print(f"\n本机IP地址已识别为: {local_ip}")

    # 为本机生成 run.sh 与 run_gpu_monitored.sh 脚本
    local_party_id = -1
    try:
        local_party_id = party_ips.index(local_ip)
        print(f"本机被识别为参与方 #{local_party_id}")
    except ValueError:
        print(f"警告: 本机IP {local_ip} 不在配置文件的 'party_ips' 列表中。")
        print("将跳过为本机生成运行脚本。")

    if local_party_id != -1:
        print("\n--- [开始处理]: 本机 ---")
        # 生成基础 run.sh
        run_command = f"./piranha -p {local_party_id} -c {config_file_path} -o output/P{local_party_id} -P 0"
        local_run_sh_path = os.path.join(script_dir, 'run.sh')
        try:
            with open(local_run_sh_path, 'w', encoding='utf-8') as f:
                f.write(run_command + '\n')
            os.chmod(local_run_sh_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            print(f"  [成功] 已在 {local_run_sh_path} 创建 run.sh 脚本。")
        except IOError as e:
            print(f"  [失败] 创建本地 run.sh 脚本失败: {e}")

        # 生成带 GPU 监控的 run_gpu_monitored.sh
        monitored_run_sh_path = os.path.join(script_dir, 'run_gpu_monitored.sh')
        monitored_script = f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="output/P{local_party_id}"
mkdir -p "$OUT_DIR"

MONITOR="monitor_gpu.py"
if [ ! -f "$MONITOR" ]; then
  echo "错误: 未找到 $MONITOR（应与本脚本同目录）。"
  exit 1
fi

read -r -p "请输入 GPU 监控记录时长（秒），直接回车表示全程记录: " DURATION
MONITOR_ARGS=()
if [[ -n "${{DURATION}}" ]]; then
  if [[ "${{DURATION}}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    MONITOR_ARGS=(-t "${{DURATION}}")
    echo "GPU 监控将记录 $DURATION 秒。"
  else
    echo "输入无效（需为数字），将默认全程记录。"
  fi
else
  echo "未输入时长，GPU 监控将全程记录。"
fi

ts="$(date +"%Y%m%d_%H%M%S")"
CSV_TMP="gpu_utilization_log.csv"
CSV_FINAL="$OUT_DIR/gpu_utilization_${{ts}}.csv"

cleanup() {{
    status=$?
    if kill -0 "${{MON_PID:-0}}" 2>/dev/null; then
        echo "发送 SIGINT 给 GPU 监控进程组..."
        kill -INT -"$MON_PID" || true
        for i in {{1..50}}; do
           if ! kill -0 "$MON_PID" 2>/dev/null; then break; fi
           sleep 0.1
        done
        if kill -0 "$MON_PID" 2>/dev/null; then
          echo "升级为 SIGTERM..."
          kill -TERM -"$MON_PID" || true
        fi
        for i in {{1..30}}; do
           if ! kill -0 "$MON_PID" 2>/dev/null; then break; fi
           sleep 0.1
        done
        if kill -0 "$MON_PID" 2>/dev/null; then
          echo "强制 KILL..."
          kill -KILL -"$MON_PID" || true
        fi
        wait "$MON_PID" 2>/dev/null || true
    fi
    if [ -f "$CSV_TMP" ]; then
        mv -f "$CSV_TMP" "$CSV_FINAL"
        echo "GPU 利用率日志已保存到 $CSV_FINAL"
    else
        echo "警告: 未找到 $CSV_TMP，可能监控未成功启动。"
    fi
    exit $status
}}
trap cleanup EXIT INT TERM

echo "启动 GPU 利用率监控..."
setsid python3 "$MONITOR" "${{MONITOR_ARGS[@]:-}}" & 
MON_PID=$!
echo "GPU 监控 PID=$MON_PID"

echo "启动 piranha..."
./piranha -p {local_party_id} -c "{config_file_path}" -o "$OUT_DIR" -P 0
"""
        try:
            with open(monitored_run_sh_path, 'w', encoding='utf-8') as f:
                f.write(monitored_script)
            os.chmod(monitored_run_sh_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            print(f"  [成功] 已在 {monitored_run_sh_path} 创建 run_gpu_monitored.sh 脚本。")
            print("        使用方式: ./run_gpu_monitored.sh（运行时会询问监控时长）")
        except IOError as e:
            print(f"  [失败] 创建本地 run_gpu_monitored.sh 脚本失败: {e}")
        print("--- [处理完毕]: 本机 ---")

    # 构建远程参与方列表
    remote_parties: List[Dict[str, any]] = []
    for i, ip in enumerate(party_ips):
        if ip != local_ip:
            user = ""
            if len(party_users) == 1:
                user = party_users[0]
            elif len(party_users) == len(party_ips):
                user = party_users[i]
            else:
                print(f"错误: 配置文件中的 'party_users' 数量 ({len(party_users)}) "
                      f"必须为1或与 'party_ips' 的数量 ({len(party_ips)}) 相等。")
                sys.exit(1)
            remote_parties.append({'id': i, 'ip': ip, 'user': user})

    if not remote_parties:
        print("\n未找到其他参与方（所有IP都与本机IP相同）。脚本执行完毕。")
        sys.exit(0)

    print(f"\n将向以下 {len(remote_parties)} 个远程参与方分发文件和脚本:")
    for party in remote_parties:
        print(f"  - 参与方 #{party['id']}: {party['user']}@{party['ip']}")

    total_steps = 4 + (1 if args.send_files and files_dir_exists else 0)

    for party in remote_parties:
        user = party['user']
        ip = party['ip']
        party_id = party['id']
        print(f"\n--- [开始处理]: 参与方 #{party_id} ({user}@{ip}) ---")

        print(f"  [1/{total_steps}] 正在远程主机上创建所需目录...")
        
        # 需要创建的目录包括 piranha 所在的目录和配置文件所在的目录
        dirs_to_create = {
            script_dir,
            config_dir
        }
        
        all_dirs_created = True
        for remote_dir in dirs_to_create:
            mkdir_cmd = ['ssh', f'{user}@{ip}', f'mkdir -p {remote_dir}']
            result = subprocess.run(mkdir_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"  [失败] 在 {ip} 上创建目录 {remote_dir} 失败。")
                print(f"  错误信息: {result.stderr.strip()}")
                all_dirs_created = False
                break
        
        if not all_dirs_created:
            continue

        print(f"  [成功] 目录已确认存在。")

        print(f"  [2/{total_steps}] 正在复制 {os.path.basename(config_file_path)} 到 {user}@{ip}:{config_file_path}")
        scp_config_cmd = ['scp', config_file_path, f'{user}@{ip}:{config_file_path}']
        result = subprocess.run(scp_config_cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print(f"  [成功] {os.path.basename(config_file_path)} 复制完成。")
        else:
            print(f"  [失败] 复制 {os.path.basename(config_file_path)} 失败。")
            print(f"  错误信息: {result.stderr.strip()}")

        print(f"  [3/{total_steps}] 正在复制 piranha 到 {user}@{ip}:{piranha_file_path}")
        scp_piranha_cmd = ['scp', piranha_file_path, f'{user}@{ip}:{piranha_file_path}']
        result = subprocess.run(scp_piranha_cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            print("  [成功] piranha 复制完成。")
        else:
            print(f"  [失败] 复制 piranha 失败。")
            print(f"  错误信息: {result.stderr.strip()}")
        
        current_step = 4
        print(f"  [{current_step}/{total_steps}] 正在为参与方 #{party_id} 创建 run.sh 脚本...")
        remote_run_sh_path = os.path.join(script_dir, 'run.sh')
        run_command = f"cd {script_dir} && ./piranha -p {party_id} -c {config_file_path} -o output/P{party_id} -P 0"
        
        ssh_run_sh_cmd_str = f'echo "{run_command}" > {remote_run_sh_path} && chmod +x {remote_run_sh_path}'
        ssh_run_sh_cmd = ['ssh', f'{user}@{ip}', ssh_run_sh_cmd_str]

        result = subprocess.run(ssh_run_sh_cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            print(f"  [成功] 在 {ip} 上创建 run.sh 脚本完成。")
        else:
            print(f"  [失败] 在 {ip} 上创建 run.sh 脚本失败。")
            print(f"  错误信息: {result.stderr.strip()}")

        if args.send_files and files_dir_exists:
            current_step += 1
            
            # --- [代码修复] ---
            # 问题：'files' 目录被发送到了远程的 config_dir。
            # 修复：将目标目录修改为 script_dir，使其与 piranha 二进制文件位于同一目录。
            remote_target_dir = script_dir
            print(f"  [{current_step}/{total_steps}] 正在使用 rsync 同步 files 目录到 {user}@{ip}:{remote_target_dir}")
            
            # rsync 命令将本地的 files_dir_path 同步到远程的 remote_target_dir (即 script_dir)
            rsync_files_cmd = ['rsync', '-avz', '--delete', files_dir_path, f'{user}@{ip}:{remote_target_dir}']
            # --- [修复结束] ---
            
            result = subprocess.run(rsync_files_cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                print("  [成功] files 目录同步完成。")
            else:
                print(f"  [失败] 同步 files 目录失败。")
                print(f"  错误信息: {result.stderr.strip()}")

        print(f"--- [处理完毕]: 参与方 #{party_id} ({user}@{ip}) ---")

    print("\n所有操作已完成。")


if __name__ == "__main__":
    main()
