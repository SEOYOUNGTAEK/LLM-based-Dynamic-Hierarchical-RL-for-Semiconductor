import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

"""
긴급 Lot 진입 유형 실험 (교수님 피드백 대응)

유형:
  early  - 투입형  : 공정 초기(Process 1) 투입 — 현행 조건 (비교 기준)
  mid    - 전환형  : 공정 중반(Process 3) 긴급 전환 — 납기 위급 전환 시나리오
  late   - 막바지형: 공정 후반(Process 5) 투입 — 잔여 공정 극소 시나리오
  mixed  - 혼합형  : early(A제품) + late(B제품) 혼재 — 실제 운영 혼재

실험 규모: 4 유형 × 2 모델(Proposed, PureDRL) × N_SEEDS = 총 runs
소요 시간: 약 3~4h/run
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

# ── 설정 ─────────────────────────────────────────────────────────────────────
SEEDS         = [42, 123, 777, 2024, 7]          # N=5
ENTRY_TYPES   = ["early", "mid", "late", "mixed"] # 4 유형
MODELS        = [
    ("proposed", "experiments/train_proposed.py"),
    ("puradrl",  "experiments/train_puradrl.py"),
]
PYTHON        = sys.executable
ROOT          = Path(__file__).parent.parent
LOG_PATH      = ROOT / "output" / "_entry_type_run_log.json"
# ─────────────────────────────────────────────────────────────────────────────

def build_run_list():
    runs = []
    for entry_type in ENTRY_TYPES:
        for seed in SEEDS:
            for model_name, script in MODELS:
                runs.append({
                    "entry_type": entry_type,
                    "seed":       seed,
                    "model":      model_name,
                    "script":     script,
                    "status":     "pending",
                    "started":    None,
                    "finished":   None,
                    "returncode": None,
                })
    return runs

def save_log(runs):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)

def load_log():
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None

def run_all(resume: bool = True):
    runs = load_log() if (resume and LOG_PATH.exists()) else build_run_list()
    save_log(runs)

    total   = len(runs)
    pending = [r for r in runs if r["status"] == "pending"]
    print(f"[{datetime.now():%H:%M}] 총 {total}개 중 {len(pending)}개 실행 예정")
    print(f"  유형: {ENTRY_TYPES}")
    print(f"  Seeds: {SEEDS}\n")

    for i, r in enumerate(runs):
        if r["status"] != "pending":
            print(f"  SKIP ({r['status']}): {r['model']} / {r['entry_type']} / s{r['seed']}")
            continue

        label = f"[{i+1}/{total}] {r['model']:12s} | type={r['entry_type']:6s} | seed={r['seed']}"
        print(f"\n{'='*60}")
        print(f"  START  {label}")
        print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")

        cmd = [
            PYTHON, r["script"],
            "--seed",              str(r["seed"]),
            "--urgent_entry_type", r["entry_type"],
        ]

        r["status"]  = "running"
        r["started"] = datetime.now().isoformat()
        save_log(runs)

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.time() - t0

        r["returncode"] = result.returncode
        r["finished"]   = datetime.now().isoformat()
        r["status"]     = "done" if result.returncode == 0 else "failed"
        save_log(runs)

        status_str = "✓ OK" if result.returncode == 0 else f"✗ FAIL(rc={result.returncode})"
        print(f"  {status_str}  {label}  ({elapsed/3600:.1f}h)")

    done   = sum(1 for r in runs if r["status"] == "done")
    failed = sum(1 for r in runs if r["status"] == "failed")
    print(f"\n{'='*60}")
    print(f"완료: {done}/{total}  실패: {failed}/{total}")
    print(f"로그: {LOG_PATH}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no_resume", action="store_true", help="이전 로그 무시하고 처음부터 실행")
    p.add_argument("--dry_run",   action="store_true", help="실제 실행 없이 계획만 출력")
    args = p.parse_args()

    if args.dry_run:
        runs = build_run_list()
        print(f"총 {len(runs)}개 실행 계획:")
        for r in runs:
            print(f"  {r['model']:12s} | type={r['entry_type']:6s} | seed={r['seed']}")
    else:
        run_all(resume=not args.no_resume)
