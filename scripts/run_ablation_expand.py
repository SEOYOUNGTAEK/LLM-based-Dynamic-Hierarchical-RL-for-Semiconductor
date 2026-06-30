import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

# run_ablation_expand.py
# Ablation 시드 확장: Soft Reward / Random Mask / Rule-Based 를 각각 +15 신규 시드로
# (기존 5 HICSS 시드 + 신규 15 = 총 20 시드, main study와 일관)
#
# - seed-major 순서: 시드 하나당 3조건을 모두 돌려 조기 부분 분석 가능
# - 이미 완료된 폴더는 skip (정규식 단어경계로 s7/s777 오탐 방지)
# - PYTHONIOENCODING=utf-8 로 progress bar 인코딩 오류 방지
# - 진행 로그 JSON 누적
#
# 실행:  .venv\Scripts\python run_ablation_expand.py
#        .venv\Scripts\python run_ablation_expand.py --dry_run

import argparse
import subprocess
import sys
import os
import re
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

OUTPUT = ROOT / "output"
LOG_PATH = ROOT / "ablation_expand_log.json"

# main study와 동일하게 총 20시드가 되도록: 기존 5 + 신규 15
NEW_SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
             1001, 1002, 1003, 1004, 1005]

# (label_regex, script, extra_args)  — label_regex 는 완료 폴더 탐지용
CONDITIONS = [
    ("Ablation_SoftReward_s{seed}", "experiments/ablations/train_ablation_soft.py",      []),
    ("Ablation_RandomMask_s{seed}", "experiments/ablations/train_ablation_randmask.py",  []),
    ("RuleBased_LLM_s{seed}_abl",   "experiments/baselines/train_baseline_rulebased.py", []),
]


def output_done(label: str) -> bool:
    """output/ 아래에 label(뒤에 숫자가 안 붙는) 폴더가 있고 결과 CSV가 있으면 완료."""
    if not OUTPUT.exists():
        return False
    pattern = re.escape(label) + r'(?!\d)'
    for d in OUTPUT.iterdir():
        if d.is_dir() and re.search(pattern, d.name):
            if list(d.glob("csv/*.csv")):
                return True
    return False


def append_log(entry: dict):
    try:
        data = json.load(open(LOG_PATH, encoding="utf-8")) if LOG_PATH.exists() else []
        data.append(entry)
        json.dump(data, open(LOG_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] log write failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # 작업 목록 (seed-major)
    jobs = []
    for seed in NEW_SEEDS:
        for label_tpl, script, extra in CONDITIONS:
            label = label_tpl.format(seed=seed)
            jobs.append((seed, label, script, ["--seed", str(seed)] + extra))

    total = len(jobs)
    print(f"[EXPAND] start {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"[EXPAND] python: {PYTHON}")
    print(f"[EXPAND] {total} jobs ({len(NEW_SEEDS)} seeds x {len(CONDITIONS)} conditions)")
    print(f"[EXPAND] est. ~2.5h/run -> ~{total*2.5:.0f}h\n")

    done = skipped = failed = 0
    for i, (seed, label, script, sargs) in enumerate(jobs, 1):
        tag = f"[{i}/{total}] {label}"
        if output_done(label):
            print(f"{tag}  -> SKIP (already done)")
            skipped += 1
            continue
        if args.dry_run:
            print(f"{tag}  -> would run: {script} {' '.join(sargs)}")
            continue

        cmd = [PYTHON, script] + sargs
        print(f"\n{tag}\n  cmd: {' '.join(cmd)}\n  start {datetime.now():%H:%M:%S}")
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
        except KeyboardInterrupt:
            print("\n[EXPAND] interrupted by user."); break
        except Exception as e:
            rc = -1; print(f"  ERROR: {e}")
        dt = (time.time() - t0) / 3600
        status = "success" if rc == 0 else "failed"
        if rc == 0: done += 1
        else: failed += 1
        print(f"  -> {status.upper()}  ({dt:.2f}h)")
        append_log({"label": label, "seed": seed, "script": script,
                    "status": status, "returncode": rc, "elapsed_h": round(dt, 2),
                    "finished_at": datetime.now().isoformat()})

    print(f"\n[EXPAND] done {datetime.now():%Y-%m-%d %H:%M:%S}  "
          f"success={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
