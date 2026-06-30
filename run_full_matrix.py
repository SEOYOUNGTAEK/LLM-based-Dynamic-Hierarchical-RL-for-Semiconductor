"""
run_full_matrix.py — clean-environment full matrix experiment
(IPLT auto-priority removed; state 21-dim, penalty -10 restored)

Matrix: condition(2) x urgent_count(3) x seed(5) = 30 runs, ~80h (3+ days)
Order: urgent_count(outer) -> seed -> condition (PureDRL+Proposed adjacent)
MongoDB/Ollama crash auto-wait; failed runs skipped and continue.

Usage:
  python run_full_matrix.py
  python run_full_matrix.py --now
  python run_full_matrix.py --quick
"""
import subprocess, sys, time, re, os, glob, socket
import pandas as pd
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

TOTAL_LOTS = 258

# urgent lot counts (even only — generator produces even counts)
URGENT_COUNTS = [20, 12, 4]          # 7.7%, 4.7%, 1.6%
SEEDS         = [42, 123, 777, 2024, 7]
NUM_EPISODES  = 100
EVAL_START    = 70

CONDITIONS = [
    ("main.py",           "PureDRL",  False),
    ("train_proposed.py", "Proposed", True),
]

if "--quick" in sys.argv:
    URGENT_COUNTS = [20, 4]
    SEEDS = [42, 123, 777]
    NUM_EPISODES = 50; EVAL_START = 35

SLEEP_BETWEEN = 30
IDLE_MINUTES  = 5
FAIL_CT       = 3000


def urgent_ratio(count):
    # generator: int(258*ratio) then //2*2 (even). (count+0.5)/258 -> exact even count.
    return (count + 0.5) / TOTAL_LOTS

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

def wait_mongo():
    for _ in range(20):
        if check_mongo(): return True
        print(f"  [wait] MongoDB down... retry in 30s [{datetime.now().strftime('%H:%M:%S')}]")
        time.sleep(30)
    return check_mongo()

def wait_for_clear():
    print("  checking for running experiments...")
    while True:
        tmp = any(f.startswith("_tmp_") and f.endswith(".py") for f in os.listdir("."))
        logs = glob.glob("output/**/*.log", recursive=True)
        idle = (time.time() - max((os.path.getmtime(l) for l in logs), default=0)) / 60
        if not tmp and idle > IDLE_MINUTES:
            print(f"  clear (idle {idle:.1f}min). starting.\n"); return
        print(f"  busy (idle {idle:.1f}min). recheck in {IDLE_MINUTES}min...")
        time.sleep(IDLE_MINUTES*60)

def patch_source(src, count, seed):
    src = re.sub(r'^(num_episodes\s*=\s*)\d+', rf'\g<1>{NUM_EPISODES}', src, flags=re.MULTILINE)
    src = re.sub(r'^(EVAL_START_EPISODE\s*=\s*)\d+', rf'\g<1>{EVAL_START}', src, flags=re.MULTILINE)
    src = re.sub(r'^(RANDOM_SEED\s*=\s*)\d+', rf'\g<1>{seed}', src, flags=re.MULTILINE)
    src = re.sub(r'generator\.run\(\s*\)', f'generator.run(urgent_lot_ratio={urgent_ratio(count)})', src)
    tag = f'_u{count}_mtx_e{NUM_EPISODES}_s{seed}'
    src = re.sub(r'(folder_name\s*=\s*f"output/\{version_no\}_PureDRL)"', rf'\1{tag}"', src)
    src = re.sub(r'(model_type\s*=\s*"[^"]*?)"', rf'\1{tag}"', src, count=1)
    return src

def run_one(script, cond, needs_llm, count, seed, idx, total):
    print(f"\n{'='*64}\n  [{idx}/{total}]  {cond}  urgent={count}  seed={seed}  {NUM_EPISODES}ep")
    print(f"  start: {datetime.now().strftime('%m-%d %H:%M:%S')}\n{'='*64}")
    if not wait_mongo():
        print("  MongoDB recovery failed - skip"); return False
    if needs_llm and not check_ollama():
        print("  Ollama down - skip"); return False
    with open(script, encoding='utf-8') as f:
        src = f.read()
    tmp = f"_tmp_mtx_{cond}_u{count}_s{seed}_{int(time.time())}.py"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(patch_source(src, count, seed))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    start = time.time(); ok = False
    errfile = f"_stderr_mtx_{cond}_u{count}_s{seed}.txt"
    try:
        with open(errfile, 'w', encoding='utf-8') as fe:
            r = subprocess.run([sys.executable, "-X", "faulthandler", tmp],
                               check=False, stderr=fe, env=env, timeout=60*60*6)
        ok = (r.returncode == 0)
        tag = "SUCCESS" if ok else f"FAILED {r.returncode}"
        print(f"\n  [{tag}]  {int(time.time()-start)//60}min")
        if not ok:
            try:
                print("  -- stderr tail --\n" + open(errfile, encoding='utf-8').read()[-700:])
            except Exception:
                pass
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
    except KeyboardInterrupt:
        if os.path.exists(tmp): os.remove(tmp)
        raise
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    if idx < total:
        time.sleep(SLEEP_BETWEEN)
    return ok

def find_csv(cond, count, seed, start_ts):
    pat = f"output/*_{cond}*_u{count}_mtx_e{NUM_EPISODES}_s{seed}/csv/results_*.csv"
    c = [p for p in glob.glob(pat) if os.path.getmtime(p) >= start_ts-60]
    return max(c, key=os.path.getmtime) if c else None

def summarize(start_ts):
    import statistics
    print(f"\n{'='*72}\n  FULL MATRIX RESULTS  ({NUM_EPISODES}ep)\n{'='*72}")
    print("  {:>7} {:<10} {:>5} {:>10} {:>9} {:>7}".format("urgent", "cond", "seeds", "CT_mean", "CT_std", "fail"))
    print("  " + "-"*54)
    table = {}
    for count in URGENT_COUNTS:
        for cond in ["PureDRL", "Proposed"]:
            cts = []
            for s in SEEDS:
                f = find_csv(cond, count, s, start_ts)
                if not f: continue
                ev = pd.read_csv(f).query("is_eval==True").AVG_ULOT_CYCLE_TIME.mean()
                cts.append(ev)
            table[(count, cond)] = cts
            if cts:
                m = statistics.mean(cts); sd = statistics.stdev(cts) if len(cts) > 1 else 0
                fails = sum(1 for x in cts if x > FAIL_CT)
                print("  {:>7} {:<10} {:>5} {:>10.0f} {:>9.0f} {:>5}/{}".format(
                    count, cond, len(cts), m, sd, fails, len(cts)))
            else:
                print("  {:>7} {:<10} {:>5}  (none)".format(count, cond, 0))
        dp = table.get((count, "PureDRL"), []); pp = table.get((count, "Proposed"), [])
        if dp and pp:
            dm = statistics.mean(dp); pm = statistics.mean(pp)
            if pm < dm - 100: eff = "LLM-better"
            elif abs(dm-pm) <= 100: eff = "similar"
            else: eff = "LLM-worse"
            print(f"          -> urgent={count}: PureDRL {dm:.0f} vs Proposed {pm:.0f}  ({(dm-pm)/dm*100:+.1f}%) [{eff}]")
        print()

def main():
    total = len(URGENT_COUNTS)*len(SEEDS)*len(CONDITIONS)
    print(f"{'='*64}")
    print(f"  FULL MATRIX (clean env, IPLT auto-priority removed)")
    print(f"  urgent={URGENT_COUNTS} x seeds={SEEDS} x cond2 = {total} runs")
    print(f"  {NUM_EPISODES}ep, est ~{total*2.7:.0f}h ({total*2.7/24:.1f} days)")
    print(f"{'='*64}\n")
    if "--now" not in sys.argv:
        wait_for_clear()
    if not check_ollama(): print("  [warn] Ollama not responding")
    if not check_mongo(): print("  [warn] MongoDB not responding (will wait at start)")

    start_ts = time.time(); idx = 0; results = []
    for count in URGENT_COUNTS:
        print(f"\n{'#'*64}\n#  urgent {count} ({count/TOTAL_LOTS*100:.1f}%) block\n{'#'*64}")
        for seed in SEEDS:
            for script, cond, needs_llm in CONDITIONS:
                idx += 1
                ok = run_one(script, cond, needs_llm, count, seed, idx, total)
                results.append((count, seed, cond, ok))
                el = int(time.time()-start_ts)
                done = sum(1 for *_, o in results if o)
                print(f"  progress {idx}/{total} (ok {done})  elapsed {el//3600}h {(el%3600)//60}m")
        summarize(start_ts)

    print(f"\n{'='*64}\n  ALL DONE\n{'='*64}")
    summarize(start_ts)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted"); sys.exit(1)
