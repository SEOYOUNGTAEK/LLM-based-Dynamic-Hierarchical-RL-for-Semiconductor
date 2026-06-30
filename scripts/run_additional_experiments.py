import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

"""
run_additional_experiments.py
─────────────────────────────
논문 추가 실험 3개를 순서대로 자동 실행

  Phase 1 — N=10 확장  (u=20, 신규 시드 5개 × PureDRL+Proposed = 10 runs, ~25h)
  Phase 2 — τ 민감도   (기존 데이터 재분석, 훈련 없음, ~1min)
  Phase 3 — Ablation   (u=20, N=5, RuleBased_LLM vs Proposed = 10 runs, ~15h)

Usage:
  python run_additional_experiments.py          # Phase 1→2→3 전체
  python run_additional_experiments.py --p1     # Phase 1만
  python run_additional_experiments.py --p2     # Phase 2만
  python run_additional_experiments.py --p3     # Phase 3만
  python run_additional_experiments.py --p1 --p3  # 1·3만
"""

import subprocess, sys, time, re, os, glob, socket, json
import pandas as pd
import statistics
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════════
#  공통 설정
# ══════════════════════════════════════════════════════════════════════
URGENT_COUNT   = 20
URGENT_RATIO   = (URGENT_COUNT + 0.5) / 258   # ≈ 0.07946
NUM_EPISODES   = 100
EVAL_START     = 70
SLEEP_BETWEEN  = 30
TIMEOUT_H      = 6
FAIL_TAU_MAIN  = 3500   # 기본 임계값

# Phase 1: N=10용 추가 시드 (기존 [42,123,777,2024,7] 제외)
EXTRA_SEEDS    = [1, 99, 314, 512, 999]

# Phase 2: τ 민감도 분석용 임계값
TAU_LIST       = [3000, 3500, 4000]

# Phase 3: Ablation 시드 (기존 N=5와 동일하게 비교)
ABLATION_SEEDS = [42, 123, 777, 2024, 7]

# ══════════════════════════════════════════════════════════════════════
#  유틸
# ══════════════════════════════════════════════════════════════════════
def ts():
    return datetime.now().strftime('%m-%d %H:%M:%S')

def sep(char='=', n=64):
    print(char * n)

def check_ollama():
    try:
        import urllib.request
        return urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3).status == 200
    except Exception:
        return False

def check_mongo():
    try:
        s = socket.create_connection(("localhost", 27017), timeout=3); s.close(); return True
    except Exception:
        return False

def wait_mongo(retries=20):
    for _ in range(retries):
        if check_mongo(): return True
        print(f"  [wait] MongoDB 재시도... [{ts()}]"); time.sleep(30)
    return check_mongo()

def patch_source(src, seed, label, tag_suffix=""):
    tag = f"_u{URGENT_COUNT}_n10_s{seed}{tag_suffix}"
    src = re.sub(r'^(num_episodes\s*=\s*)\d+',       rf'\g<1>{NUM_EPISODES}', src, flags=re.MULTILINE)
    src = re.sub(r'^(EVAL_START_EPISODE\s*=\s*)\d+',  rf'\g<1>{EVAL_START}',  src, flags=re.MULTILINE)
    src = re.sub(r'^(RANDOM_SEED\s*=\s*)\d+',         rf'\g<1>{seed}',        src, flags=re.MULTILINE)
    src = re.sub(r'generator\.run\(\s*\)',
                 f'generator.run(urgent_lot_ratio={URGENT_RATIO})', src)
    src = re.sub(r'(folder_name\s*=\s*f"output/\{version_no\}_PureDRL)"',
                 rf'\1{tag}"', src)
    src = re.sub(r'(model_type\s*=\s*"[^"]*?)"', rf'\1{tag}"', src, count=1)
    return src

def run_one(script, label, needs_llm, seed, idx, total, tag_suffix="", phase_label=""):
    sep()
    print(f"  {phase_label}[{idx}/{total}]  {label}  seed={seed}  {NUM_EPISODES}ep  [{ts()}]")
    sep()
    if not wait_mongo():
        print("  MongoDB 복구 실패 — skip"); return False
    if needs_llm and not check_ollama():
        print("  Ollama 응답 없음 — skip"); return False

    with open(script, encoding='utf-8') as f:
        src = f.read()
    tmp = f"_tmp_add_{label}_u{URGENT_COUNT}_s{seed}_{int(time.time())}.py"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(patch_source(src, seed, label, tag_suffix))

    errfile = f"_stderr_add_{label}_u{URGENT_COUNT}_s{seed}.txt"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    start = time.time(); ok = False
    try:
        with open(errfile, 'w', encoding='utf-8') as fe:
            r = subprocess.run(
                [sys.executable, "-X", "faulthandler", tmp],
                check=False, stderr=fe, env=env, timeout=TIMEOUT_H * 3600)
        ok = (r.returncode == 0)
        elapsed = int(time.time() - start)
        tag = "SUCCESS ✅" if ok else f"FAILED ❌ (code {r.returncode})"
        print(f"\n  [{tag}]  {elapsed//3600}h {(elapsed%3600)//60}m")
        if not ok:
            try:
                print("  -- stderr tail --\n" + open(errfile, encoding='utf-8').read()[-600:])
            except Exception:
                pass
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({TIMEOUT_H}h) ❌")
    except KeyboardInterrupt:
        if os.path.exists(tmp): os.remove(tmp)
        raise
    finally:
        if os.path.exists(tmp): os.remove(tmp)

    if idx < total:
        print(f"  다음까지 {SLEEP_BETWEEN}s 대기...")
        time.sleep(SLEEP_BETWEEN)
    return ok


# ══════════════════════════════════════════════════════════════════════
#  Phase 1: N=10 확장
# ══════════════════════════════════════════════════════════════════════
def phase1_n10():
    sep('#')
    print(f"  PHASE 1 — N=10 확장  (u={URGENT_COUNT}, 신규 시드 {EXTRA_SEEDS})")
    print(f"  PureDRL × {len(EXTRA_SEEDS)}  +  Proposed × {len(EXTRA_SEEDS)}  =  {len(EXTRA_SEEDS)*2} runs")
    sep('#')

    runs = []
    for seed in EXTRA_SEEDS:
        runs.append(("main.py",           "PureDRL",         False, seed))
        runs.append(("train_proposed.py", "Proposed_FullLLM", True,  seed))

    results = []
    total = len(runs)
    for idx, (script, label, needs_llm, seed) in enumerate(runs, 1):
        ok = run_one(script, label, needs_llm, seed, idx, total, phase_label="P1-")
        results.append((label, seed, ok))
        done = sum(1 for *_, o in results if o)
        print(f"  진행 {idx}/{total}  (성공 {done})\n")

    sep()
    print("  Phase 1 완료 요약")
    sep()
    for label, seed, ok in results:
        print(f"  {label:<22} seed={seed}  {'✅' if ok else '❌'}")
    return results


# ══════════════════════════════════════════════════════════════════════
#  Phase 2: τ 민감도 분석 (재훈련 없음)
# ══════════════════════════════════════════════════════════════════════
def phase2_tau_sensitivity():
    sep('#')
    print(f"  PHASE 2 — τ 민감도 분석  (τ = {TAU_LIST})")
    print(f"  기존 u=20 mtx_e100 결과 재분석 (훈련 없음)")
    sep('#')

    # 기존 + N=10 결과 전체 수집
    all_data = {}   # (cond, seed) -> [eval CTs]
    patterns = glob.glob("output/*_u20_*e100*/csv/results_*.csv") + \
               glob.glob("output/*_u20_*e100*/csv/rewards_data.csv") + \
               glob.glob("output/*_u20_*n10*/csv/results_*.csv")

    for path in patterns:
        try:
            df = pd.read_csv(path)
            df_eval = df[df['is_eval'] == True]
            if df_eval.empty: continue
            cond = "Proposed" if "Proposed" in path or "n10" in path and "PureDRL" not in path else "PureDRL"
            # 좀 더 정확하게
            if "PureDRL" in path: cond = "PureDRL"
            elif "Proposed" in path: cond = "Proposed"
            seed = int(df_eval['seed'].iloc[0])
            key = (cond, seed)
            if key not in all_data:
                all_data[key] = []
            all_data[key].extend(df_eval['AVG_ULOT_CYCLE_TIME'].tolist())
        except Exception as e:
            print(f"  [skip] {path}: {e}")

    # τ별 분석
    tau_results = {}
    sep()
    print(f"  {'τ':>5}  {'조건':<12}  {'시드수':>5}  {'실패시드':>7}  {'실패율':>6}  {'평균CT':>8}")
    sep('-')
    for tau in TAU_LIST:
        tau_results[tau] = {}
        for cond in ["PureDRL", "Proposed"]:
            seed_results = []
            for (c, s), cts in all_data.items():
                if c != cond: continue
                avg_ct = statistics.mean(cts)
                seed_results.append((s, avg_ct, avg_ct > tau))
            fails = sum(1 for *_, f in seed_results if f)
            n = len(seed_results)
            avg_ct_all = statistics.mean([ct for _, ct, _ in seed_results]) if seed_results else 0
            fail_rate = fails/n*100 if n > 0 else 0
            tau_results[tau][cond] = {'n': n, 'fails': fails, 'fail_rate': fail_rate}
            print(f"  {tau:>5}  {cond:<12}  {n:>5}  {fails:>7}  {fail_rate:>5.0f}%  {avg_ct_all:>8.0f}")
        # 개선율
        pur = tau_results[tau].get("PureDRL", {})
        pro = tau_results[tau].get("Proposed", {})
        if pur and pro and pur['fail_rate'] > 0:
            improvement = pur['fail_rate'] - pro['fail_rate']
            print(f"         → τ={tau}: 실패율 개선 {improvement:+.0f}%p  "
                  f"({'완전 제거' if pro['fail_rate']==0 else '부분 개선'})")
        print()

    # 결과 저장
    out = {"tau_sensitivity": tau_results, "generated_at": ts()}
    os.makedirs("output/_analysis", exist_ok=True)
    with open("output/_analysis/tau_sensitivity.json", 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  결과 저장: output/_analysis/tau_sensitivity.json")
    return tau_results


# ══════════════════════════════════════════════════════════════════════
#  Phase 3: Ablation — RuleBased(고정룰, LLM없음) vs Proposed
# ══════════════════════════════════════════════════════════════════════
def phase3_ablation():
    sep('#')
    print(f"  PHASE 3 — Ablation  (u={URGENT_COUNT}, seeds={ABLATION_SEEDS})")
    print(f"  RuleBased_LLM(LLM없음) × {len(ABLATION_SEEDS)}  =  {len(ABLATION_SEEDS)} runs")
    print(f"  (Proposed 기존 결과와 비교 → LLM의 가치 검증)")
    sep('#')

    results = []
    total = len(ABLATION_SEEDS)
    for idx, seed in enumerate(ABLATION_SEEDS, 1):
        ok = run_one("train_baseline_rulebased.py", "RuleBased_LLM", False,
                     seed, idx, total, tag_suffix="_abl", phase_label="P3-")
        results.append(("RuleBased_LLM", seed, ok))
        done = sum(1 for *_, o in results if o)
        print(f"  진행 {idx}/{total}  (성공 {done})\n")

    sep()
    print("  Phase 3 완료 요약")
    sep()
    for label, seed, ok in results:
        print(f"  {label:<22} seed={seed}  {'✅' if ok else '❌'}")

    # 기존 PureDRL / Proposed와 비교 요약
    sep()
    print("  Ablation 비교 (기존 u=20 mtx_e100 데이터 포함)")
    sep('-')
    for cond_label, pattern in [
        ("PureDRL",      "output/*PureDRL_u20_mtx_e100*/csv/results_*.csv"),
        ("Proposed",     "output/*Proposed*u20_mtx_e100*/csv/results_*.csv"),
        ("RuleBased",    "output/*RuleBased*u20*n10*abl*/csv/results_*.csv"),
    ]:
        cts = []
        for path in glob.glob(pattern):
            try:
                df = pd.read_csv(path)
                ev = df[df['is_eval'] == True]['AVG_ULOT_CYCLE_TIME'].mean()
                cts.append(ev)
            except Exception:
                pass
        if cts:
            m = statistics.mean(cts)
            fails = sum(1 for x in cts if x > FAIL_TAU_MAIN)
            print(f"  {cond_label:<14} N={len(cts)}  AvgCT={m:.0f}  Fail={fails}/{len(cts)} ({fails/len(cts)*100:.0f}%)")
    return results


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run_p1 = "--p1" in sys.argv or not any(a in sys.argv for a in ["--p1","--p2","--p3"])
    run_p2 = "--p2" in sys.argv or not any(a in sys.argv for a in ["--p1","--p2","--p3"])
    run_p3 = "--p3" in sys.argv or not any(a in sys.argv for a in ["--p1","--p2","--p3"])

    sep('=')
    print(f"  run_additional_experiments.py  [{ts()}]")
    print(f"  실행 Phase:  {'P1(N=10) ' if run_p1 else ''}{'P2(τ분석) ' if run_p2 else ''}{'P3(Ablation)' if run_p3 else ''}")
    print(f"  u={URGENT_COUNT}  τ={FAIL_TAU_MAIN}s  기본 에피소드={NUM_EPISODES}")
    sep('=')
    print()

    start_total = time.time()
    all_ok = True

    if run_p1:
        r1 = phase1_n10()
        if not all(ok for *_, ok in r1):
            print("  ⚠️  Phase 1 일부 실패 — Phase 2/3 계속 진행합니다\n")
            all_ok = False

    if run_p2:
        phase2_tau_sensitivity()

    if run_p3:
        r3 = phase3_ablation()
        if not all(ok for *_, ok in r3):
            all_ok = False

    elapsed = int(time.time() - start_total)
    sep('=')
    print(f"  전체 완료  {'✅' if all_ok else '⚠️ (일부 실패)'}  "
          f"총 {elapsed//3600}h {(elapsed%3600)//60}m  [{ts()}]")
    sep('=')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨"); sys.exit(1)
