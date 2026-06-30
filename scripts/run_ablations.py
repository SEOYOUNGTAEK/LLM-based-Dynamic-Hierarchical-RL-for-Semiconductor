import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

# run_ablations.py
# AAAI 실험 전체 순차 자동 실행 스크립트
#
# Phase 1 — Ablation (30개, 먼저 실행)
#   Ablation 1 — Hard Mask vs Soft Reward        (5 seeds)
#   Ablation 2 — LLM vs Random Rules             (5 seeds)
#   Ablation 3 — Dynamic vs Static Rules         (5 seeds × 3 variants)
#   Ablation 4 — Random Mask                     (5 seeds)
#
# Phase 2 — Seed 확장 (나중 실행, 중단 후 재시작 가능)
#   PureDRL  × 15 new seeds  (HICSS 5 + 확장 15 = 총 20 seeds)
#   LLM-HRL  × 15 new seeds
#
# 사용법:
#   .venv\Scripts\python run_ablations.py              # 전체 (ablation → seed 확장)
#   .venv\Scripts\python run_ablations.py --dry_run    # 목록만 출력
#   .venv\Scripts\python run_ablations.py --only_abl 1 # 특정 ablation만
#   .venv\Scripts\python run_ablations.py --phase 1    # ablation만
#   .venv\Scripts\python run_ablations.py --phase 2    # seed 확장만

import argparse
import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# ── 공통 설정 ────────────────────────────────────────────────────
PYTHON = str(Path(__file__).parent / ".venv" / "Scripts" / "python.exe")
if not Path(PYTHON).exists():
    PYTHON = sys.executable

TIMEOUT_SEC = 12 * 3600  # 실험당 최대 12시간

# Ablation용 seeds — HICSS와 동일 (기존 proposed 결과와 직접 비교 가능)
SEEDS = [42, 123, 777, 2024, 7]

# Seed 확장 — HICSS 5개 이후 새로운 15개 (총 20 seeds)
# HICSS seeds는 output 폴더가 없으므로 skip되지 않음 → 필요 시 수동 skip
EXPANSION_NEW_SEEDS = [100, 200, 300, 400, 500,
                       600, 700, 800, 900, 1000,
                       1001, 1002, 1003, 1004, 1005]


def build_experiments(only_abl=None):
    """실험 리스트 생성. (label, script, args) 튜플 목록 반환."""
    exps = []

    # ── Ablation 1: Hard Mask vs Soft Reward ──────────────────
    if only_abl in (None, 1):
        for seed in SEEDS:
            exps.append({
                "ablation": 1,
                "label":    f"Ablation_SoftReward_s{seed}",
                "script":   "experiments/ablations/train_ablation_soft.py",
                "args":     ["--seed", str(seed)],
            })

    # ── Ablation 2: LLM vs Random Rules ───────────────────────
    if only_abl in (None, 2):
        for seed in SEEDS:
            exps.append({
                "ablation": 2,
                "label":    f"Ablation_RandomRules_s{seed}",
                "script":   "experiments/ablations/train_ablation_random.py",
                "args":     ["--seed", str(seed)],
            })

    # ── Ablation 3a: Static — every 5 episodes ────────────────
    if only_abl in (None, 3):
        for seed in SEEDS:
            exps.append({
                "ablation": 3,
                "label":    f"Ablation_StaticRules_Every5_s{seed}",
                "script":   "experiments/ablations/train_ablation_static.py",
                "args":     ["--seed", str(seed), "--update_interval", "5"],
            })

    # ── Ablation 3b: Static — fixed (never update after ep1) ──
    if only_abl in (None, 3):
        for seed in SEEDS:
            exps.append({
                "ablation": 3,
                "label":    f"Ablation_StaticRules_Fixed_s{seed}",
                "script":   "experiments/ablations/train_ablation_static.py",
                "args":     ["--seed", str(seed), "--update_interval", "0"],
            })

    # ── Ablation 3c: Static — every 10 episodes ───────────────
    if only_abl in (None, 3):
        for seed in SEEDS:
            exps.append({
                "ablation": 3,
                "label":    f"Ablation_StaticRules_Every10_s{seed}",
                "script":   "experiments/ablations/train_ablation_static.py",
                "args":     ["--seed", str(seed), "--update_interval", "10"],
            })

    # ── Ablation 4: Random Mask (조건 없는 에피소드별 랜덤 차단) ──
    if only_abl in (None, 4):
        for seed in SEEDS:
            exps.append({
                "ablation": 4,
                "label":    f"Ablation_RandomMask_s{seed}",
                "script":   "experiments/ablations/train_ablation_randmask.py",
                "args":     ["--seed", str(seed)],
            })

    return exps


def build_seed_expansion() -> list:
    """
    Phase 2: Seed 확장 실험 목록.
    PureDRL + LLM-HRL을 새 15 seeds로 실행.
    HICSS 5 seeds는 이미 있으므로 총 20 seeds가 됨.
    실험 순서: PureDRL(s100) → LLM-HRL(s100) → PureDRL(s200) → ...
    (같은 seed끼리 붙여서 비교 편의성 확보)
    """
    exps = []
    for seed in EXPANSION_NEW_SEEDS:
        exps.append({
            "ablation": "expand_drl",
            "label":    f"PureDRL_s{seed}",
            "script":   "main.py",
            "args":     ["--seed", str(seed)],
        })
        exps.append({
            "ablation": "expand_llm",
            "label":    f"Proposed_FullLLM_s{seed}",
            "script":   "experiments/train_proposed.py",
            "args":     ["--seed", str(seed)],
        })
    return exps


def output_dir_exists(label: str) -> bool:
    """output/ 아래에 label을 포함하는 폴더가 있으면 이미 완료된 것으로 간주.
    label 매칭은 단어 경계 기준: 폴더명에서 label 뒤에 숫자가 이어지는 오탐 방지.
    예: label='Ablation_SoftReward_s7'이 '_s777' 폴더에 오탐되지 않도록.
    """
    import re
    output_root = Path(__file__).parent / "output"
    if not output_root.exists():
        return False
    # label 뒤에 숫자가 오지 않는 경우만 매칭 (단어 끝 또는 non-digit)
    pattern = re.escape(label) + r'(?!\d)'
    for d in output_root.iterdir():
        if d.is_dir() and re.search(pattern, d.name):
            csv_files = list(d.glob("csv/results_*.csv"))
            if csv_files:
                return True
    return False


def run_experiment(exp: dict, log_path: str, dry_run: bool = False) -> dict:
    """단일 실험 실행. 결과 dict 반환."""
    label  = exp["label"]
    script = exp["script"]
    args   = exp["args"]
    cmd    = [PYTHON, script] + args

    result = {
        "label":      label,
        "ablation":   exp["ablation"],
        "script":     script,
        "args":       args,
        "started_at": datetime.now().isoformat(),
        "status":     None,
        "elapsed_sec": None,
        "returncode": None,
        "error":      None,
    }

    print(f"\n{'='*60}")
    print(f"[RUN] {label}")
    print(f"      cmd: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        result["status"] = "dry_run"
        return result

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            timeout=TIMEOUT_SEC,
            cwd=str(Path(__file__).parent),
            # stdout/stderr를 터미널에 그대로 흘려보냄 (진행 바 표시)
        )
        elapsed = time.time() - t0
        result["elapsed_sec"] = round(elapsed, 1)
        result["returncode"]  = proc.returncode
        if proc.returncode == 0:
            result["status"] = "success"
            print(f"\n[OK]  {label}  ({elapsed/3600:.1f}h)")
        else:
            result["status"] = "failed_nonzero"
            result["error"]  = f"returncode={proc.returncode}"
            print(f"\n[FAIL] {label}  returncode={proc.returncode}")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        result["status"]      = "timeout"
        result["elapsed_sec"] = round(elapsed, 1)
        result["error"]       = f"Timeout after {TIMEOUT_SEC}s"
        print(f"\n[TIMEOUT] {label}  exceeded {TIMEOUT_SEC/3600:.0f}h limit")

    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["error"]  = "KeyboardInterrupt"
        print(f"\n[INTERRUPTED] {label}")
        raise  # 상위로 전파 → 전체 루프 중단

    except Exception as e:
        elapsed = time.time() - t0
        result["status"]      = "exception"
        result["elapsed_sec"] = round(elapsed, 1)
        result["error"]       = str(e)
        print(f"\n[ERROR] {label}: {e}")

    result["finished_at"] = datetime.now().isoformat()

    # 로그 누적 저장
    _append_log(log_path, result)
    return result


def _append_log(log_path: str, entry: dict):
    """실험 결과를 JSON 배열에 누적 저장."""
    try:
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to write log: {e}")


def print_summary(results: list):
    """전체 실행 결과 요약 출력."""
    total   = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed  = sum(1 for r in results if r["status"] not in ("success", "skipped", "dry_run"))
    skipped = sum(1 for r in results if r["status"] == "skipped")

    print("\n" + "="*60)
    print("ABLATION RUN SUMMARY")
    print("="*60)
    print(f"  Total    : {total}")
    print(f"  Success  : {success}")
    print(f"  Skipped  : {skipped}")
    print(f"  Failed   : {failed}")
    print()

    for r in results:
        status_icon = {"success": "✓", "skipped": "–", "dry_run": "○"}.get(r["status"], "✗")
        elapsed = f"  {r['elapsed_sec']/3600:.1f}h" if r.get("elapsed_sec") else ""
        print(f"  {status_icon} {r['label']}{elapsed}")
        if r.get("error"):
            print(f"      → {r['error']}")

    print("="*60)


def run_experiment_list(experiments, log_path, skip_done, dry_run):
    """experiments 목록을 순서대로 실행. 결과 목록 반환."""
    results = []
    total = len(experiments)
    try:
        for i, exp in enumerate(experiments):
            label = exp["label"]
            print(f"\n[{i+1}/{total}] {label}")

            if skip_done and not dry_run and output_dir_exists(label):
                print(f"  → SKIPPED (output folder found)")
                results.append({**exp, "status": "skipped", "error": None,
                                "elapsed_sec": None, "returncode": None})
                _append_log(log_path, results[-1])
                continue

            result = run_experiment(exp, log_path, dry_run=dry_run)
            results.append(result)

            if result["status"] not in ("success", "dry_run", "skipped"):
                print(f"  → Continuing with next experiment despite failure...")

    except KeyboardInterrupt:
        print("\n\n[RUNNER] Interrupted by user. Saving partial results...")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AAAI Ablation sequential runner")
    parser.add_argument("--dry_run",  action="store_true",
                        help="목록만 출력, 실제 실행 없음")
    parser.add_argument("--only_abl", type=int, default=None,
                        choices=[1, 2, 3, 4],
                        help="특정 ablation만 실행 (1=Soft/2=RandomRule/3=Static/4=RandomMask). Phase 1에서만 유효.")
    parser.add_argument("--phase", type=int, default=0,
                        choices=[0, 1, 2],
                        help="0=전체(Phase1→Phase2), 1=Ablation만, 2=Seed확장만")
    parser.add_argument("--no_skip",   action="store_true",
                        help="이미 완료된 실험도 강제 재실행")
    args = parser.parse_args()

    skip_done = not args.no_skip
    run_phase1 = args.phase in (0, 1)
    run_phase2 = args.phase in (0, 2)

    log_path = str(Path(__file__).parent / "ablation_run_log.json")

    print(f"[RUNNER] Python   : {PYTHON}")
    print(f"[RUNNER] Timeout  : {TIMEOUT_SEC/3600:.0f}h per experiment")
    print(f"[RUNNER] Skip done: {skip_done}")
    print(f"[RUNNER] Phase    : {'both' if args.phase==0 else ('ablation only' if args.phase==1 else 'seed expansion only')}")

    all_results = []

    # ── Phase 1: Ablation ────────────────────────────────────────
    if run_phase1:
        abl_exps = build_experiments(only_abl=args.only_abl)
        print(f"\n{'='*60}")
        print(f"PHASE 1 - ABLATION  ({len(abl_exps)} experiments)")
        print(f"{'='*60}")
        results1 = run_experiment_list(abl_exps, log_path, skip_done, args.dry_run)
        all_results.extend(results1)
        print_summary(results1)

    # ── Phase 2: Seed Expansion ──────────────────────────────────
    if run_phase2:
        exp_exps = build_seed_expansion()
        print(f"\n{'='*60}")
        print(f"PHASE 2 - SEED EXPANSION  ({len(exp_exps)} experiments)")
        print(f"  PureDRL x {len(EXPANSION_NEW_SEEDS)} new seeds")
        print(f"  LLM-HRL x {len(EXPANSION_NEW_SEEDS)} new seeds")
        print(f"  (HICSS 5 seeds are NOT included - already done)")
        print(f"{'='*60}")
        results2 = run_experiment_list(exp_exps, log_path, skip_done, args.dry_run)
        all_results.extend(results2)
        print_summary(results2)

    if run_phase1 and run_phase2:
        print("\n" + "="*60)
        print("OVERALL SUMMARY (Phase 1 + Phase 2)")
        print_summary(all_results)

    print(f"\nFull log: {log_path}")
