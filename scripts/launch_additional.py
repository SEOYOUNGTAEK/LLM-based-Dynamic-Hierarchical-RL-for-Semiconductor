import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

"""
launch_additional.py
────────────────────
현재 진행 중인 실험(_tmp_*.py)이 모두 끝나면
run_additional_experiments.py 를 자동으로 시작한다.
"""
import os, glob, time, subprocess, sys
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def ts():
    return datetime.now().strftime('%H:%M:%S')

print(f"[{ts()}] 현재 실행 중인 실험 대기 중...")
while True:
    tmp = glob.glob("_tmp_*.py")
    if not tmp:
        print(f"[{ts()}] 모든 실험 완료 — run_additional_experiments.py 시작!\n")
        break
    logs = glob.glob("output/*/log/*.log")
    if logs:
        latest = max(logs, key=os.path.getmtime)
        folder = latest.split(os.sep)[1]
        last_lines = open(latest, encoding='utf-8', errors='ignore').readlines()
        ep_lines = [l.strip() for l in last_lines if 'Episode' in l]
        status = ep_lines[-1] if ep_lines else "진행 중..."
        print(f"[{ts()}] {folder} — {status}")
    time.sleep(120)  # 2분마다 체크

subprocess.run([sys.executable, "run_additional_experiments.py"])
