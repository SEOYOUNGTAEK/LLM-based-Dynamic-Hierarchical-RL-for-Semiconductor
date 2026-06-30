import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

"""
sweep_urgent.py — Urgent Lot 희소도(sparsity) 스윕 실험
────────────────────────────────────────────────────────────
목적:
  Urgent lot 개수를 줄여 "sparse reward" 상황을 만들고,
  각 희소도에서 PureDRL(실패 예상) vs Proposed_LLM(보완 예상)을 비교.
  → "DRL은 못 배우는데 LLM이 보완 → CT 개선 + 빠른 수렴" 스위트스팟 탐색.

사용법:
  python sweep_urgent.py            # 기본 (현재 실행 끝나길 기다린 뒤 시작)
  python sweep_urgent.py --now      # 대기 없이 즉시 시작 (주의: 다른 실행과 충돌 가능)

설계:
  - URGENT_COUNTS = [4, 6, 8, 2]  (가능성 높은 순서, 극단은 마지막)
  - 각 개수마다 PureDRL → Proposed 순서 (중간 중단돼도 비교쌍 확보)
  - 에피소드 40 (train 30 + eval 10) — 스윕용 경량
  - 결과 폴더명에 _u{count} 태그 → 아침에 식별 쉬움
  - 종료 후 자동으로 요약 테이블 출력

밤새 돌려놓고 아침에 결과 보기용.
"""

import subprocess
import sys
import time
import re
import os
import glob
import pandas as pd
from datetime import datetime, timedelta

# 한국어 콘솔(cp949)에서 이모지 출력 시 UnicodeEncodeError 방지
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# ════════════════════════════════════════════════════════════════
# 설정
# ════════════════════════════════════════════════════════════════
URGENT_COUNTS = [4, 6, 8, 2]      # 스윕할 urgent lot 개수 (현재 기본 20)
TOTAL_LOTS    = 258               # data_generator.py와 동일

NUM_EPISODES  = 40                # train 30 + eval 10 (스윕용 경량)
EVAL_START    = 30

# (스크립트, 조건이름, LLM필요여부)
CONDITIONS = [
    ("main.py",           "PureDRL",  False),   # LLM 없음 — 실패 예상
    ("train_proposed.py", "Proposed", True),    # Full LLM — 보완 예상
]

SLEEP_BETWEEN = 30                # 실험 간 대기 (메모리/Plantsim 정리)
IDLE_MINUTES  = 5                 # 이 시간 동안 로그 변화 없으면 "현재 실행 끝남"으로 판단


# ════════════════════════════════════════════════════════════════
# 유틸리티
# ════════════════════════════════════════════════════════════════

def urgent_ratio(count: int) -> float:
    """count개 urgent lot이 정확히 나오는 ratio.
    generator: target_total = int(258*ratio); per_device = target_total//2; total = per_device*2
    ratio=(count+0.5)/258 → 258*ratio = count+0.5 → int = count → 정확히 count개 보장."""
    return (count + 0.5) / TOTAL_LOTS


def check_ollama() -> bool:
    try:
        import urllib.request
        r = urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3)
        return r.status == 200
    except Exception:
        return False


def wait_for_clear():
    """현재 다른 학습이 돌고 있으면 끝날 때까지 대기."""
    print("  현재 실행 중인 학습이 있는지 확인...")
    while True:
        # _tmp_*.py (run_llm_only/run_all이 만드는 임시 파일) 존재 여부
        tmp_running = any(f.startswith("_tmp_") and f.endswith(".py")
                          for f in os.listdir("."))
        # 가장 최근 로그 수정 시각
        logs = glob.glob("output/**/*.log", recursive=True)
        newest = max((os.path.getmtime(l) for l in logs), default=0)
        idle_min = (time.time() - newest) / 60 if newest else 999

        if not tmp_running and idle_min > IDLE_MINUTES:
            print(f"  ✅ 현재 실행 없음 (로그 idle {idle_min:.1f}분). 스윕 시작.\n")
            return
        print(f"  ⏳ 실행 중 (tmp={tmp_running}, idle={idle_min:.1f}분). "
              f"{IDLE_MINUTES}분 후 재확인... [{datetime.now().strftime('%H:%M:%S')}]")
        time.sleep(IDLE_MINUTES * 60)


def patch_source(src: str, ratio: float, count: int) -> str:
    """에피소드 수, urgent ratio, 결과 폴더 태그 패치."""
    src = re.sub(r'^(num_episodes\s*=\s*)\d+',
                 rf'\g<1>{NUM_EPISODES}', src, flags=re.MULTILINE)
    src = re.sub(r'^(EVAL_START_EPISODE\s*=\s*)\d+',
                 rf'\g<1>{EVAL_START}', src, flags=re.MULTILINE)
    # generator.run() → generator.run(urgent_lot_ratio=...)
    src = re.sub(r'generator\.run\(\s*\)',
                 f'generator.run(urgent_lot_ratio={ratio})', src)
    # 폴더 태그: main.py(folder_name 하드코딩) + train_proposed(model_type)
    src = re.sub(r'(folder_name\s*=\s*f"output/\{version_no\}_PureDRL)"',
                 rf'\1_u{count}"', src)
    src = re.sub(r'(model_type\s*=\s*"[^"]*?)"',
                 rf'\1_u{count}"', src, count=1)
    return src


def run_one(script: str, cond: str, needs_llm: bool, count: int,
            idx: int, total: int) -> bool:
    ratio = urgent_ratio(count)
    print(f"\n{'='*64}")
    print(f"  [{idx}/{total}]  {cond}  |  urgent={count}개 (ratio={ratio:.5f})")
    print(f"  {script}  |  {NUM_EPISODES}ep (train={EVAL_START}, eval={NUM_EPISODES-EVAL_START})")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*64)

    if needs_llm and not check_ollama():
        print("  ❌ Ollama 응답 없음 — 이 조건 건너뜀")
        return False

    with open(script, encoding='utf-8') as f:
        src = f.read()
    patched = patch_source(src, ratio, count)

    tmp = f"_tmp_sweep_{cond}_u{count}_{int(time.time())}.py"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(patched)

    stderr_path = f"_stderr_sweep_{cond}_u{count}.txt"
    child_env = dict(os.environ, PYTHONIOENCODING="utf-8")
    start = time.time()
    ok = False
    try:
        with open(stderr_path, 'w', encoding='utf-8') as ferr:
            r = subprocess.run([sys.executable, "-X", "faulthandler", tmp],
                               check=False, stderr=ferr, env=child_env,
                               timeout=60*60*5)
        ok = (r.returncode == 0)
        status = "✅ SUCCESS" if ok else f"❌ FAILED ({r.returncode})"
        if not ok:
            try:
                with open(stderr_path, encoding='utf-8') as ferr:
                    print(f"\n  ── stderr 마지막 1500자 ──\n{ferr.read()[-1500:]}\n  ──")
            except Exception:
                pass
    except subprocess.TimeoutExpired:
        status = "⏰ TIMEOUT"
    except KeyboardInterrupt:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    el = int(time.time() - start)
    print(f"\n  {status}  |  소요: {el//3600}h {(el%3600)//60}m")
    if idx < total:
        print(f"  ⏳ {SLEEP_BETWEEN}초 대기...")
        time.sleep(SLEEP_BETWEEN)
    return ok


def summarize(sweep_start_ts: float):
    """스윕 결과 폴더들을 읽어 요약 테이블 출력."""
    print(f"\n{'='*72}")
    print("  스윕 결과 요약 (eval 구간)")
    print('='*72)
    rows = []
    for count in URGENT_COUNTS:
        for script, cond, _ in CONDITIONS:
            # _u{count} 태그 폴더 중 스윕 시작 이후 생성된 것
            pat = f"output/*_u{count}/csv/results_*.csv"
            cands = [p for p in glob.glob(pat)
                     if os.path.getmtime(p) >= sweep_start_ts - 60]
            # 조건 이름으로 필터
            cands = [p for p in cands if cond.split('_')[0] in p]
            if not cands:
                rows.append((count, cond, None))
                continue
            path = max(cands, key=os.path.getmtime)
            d = pd.read_csv(path)
            ev = d[d.is_eval == True] if 'is_eval' in d.columns else d
            if len(ev) == 0:
                ev = d
            rows.append((count, cond, ev))

    hdr = "{:>7} {:<10} {:>5} {:>9} {:>8} {:>9} {:>8}"
    print(hdr.format("urgent", "cond", "n", "CT_mean", "CT_std", "TM_mean", "TM_std"))
    print("-"*64)
    for count, cond, ev in rows:
        if ev is None:
            print(hdr.format(count, cond, 0, "-", "-", "-", "-"))
            continue
        print(hdr.format(count, cond, len(ev),
            f"{ev.AVG_ULOT_CYCLE_TIME.mean():.0f}",
            f"{ev.AVG_ULOT_CYCLE_TIME.std(ddof=1):.0f}",
            f"{ev.target_meet_rate.mean():.2f}",
            f"{ev.target_meet_rate.std(ddof=1):.2f}"))

    print()
    print("  해석 가이드:")
    print("   - PureDRL CT가 높고 CT_std 큼 + Proposed CT 낮고 안정 → 그 urgent 개수가 스위트스팟")
    print("   - 두 조건 CT 비슷 → 그 개수는 아직 DRL이 학습 가능 (덜 sparse)")


def main():
    print("="*64)
    print("  Urgent Lot Sparsity 스윕 실험")
    print(f"  개수: {URGENT_COUNTS}  ×  조건: {[c[1] for c in CONDITIONS]}")
    print(f"  에피소드: {NUM_EPISODES} (train {EVAL_START} + eval {NUM_EPISODES-EVAL_START})")
    total = len(URGENT_COUNTS) * len(CONDITIONS)
    print(f"  총 {total}개 run")
    print("="*64 + "\n")

    if "--now" not in sys.argv:
        wait_for_clear()

    if not check_ollama():
        print("  ⚠️  Ollama 응답 없음. Proposed 조건은 건너뜁니다. (ollama serve 필요)")

    sweep_start = time.time()
    results = []
    idx = 0
    # 개수별로 PureDRL → Proposed (중단돼도 비교쌍 확보)
    for count in URGENT_COUNTS:
        for script, cond, needs_llm in CONDITIONS:
            idx += 1
            ok = run_one(script, cond, needs_llm, count, idx, total)
            results.append((count, cond, ok))
            done = sum(1 for *_, o in results if o)
            el = int(time.time() - sweep_start)
            print(f"  진행: {idx}/{total} (성공 {done})  |  경과 {el//3600}h {(el%3600)//60}m")

    print(f"\n{'='*64}")
    print("  전체 완료 요약")
    print('='*64)
    for count, cond, ok in results:
        print(f"  {'✅' if ok else '❌'}  u{count}  {cond}")

    summarize(sweep_start)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ⛔ 중단됨")
        sys.exit(1)
