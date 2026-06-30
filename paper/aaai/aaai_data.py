"""
aaai_data.py
────────────────────────────────────────────────────────────────────────────
Central data loader for all AAAI-27 figures.
Pulls evaluation cycle-time data directly from output/ run folders so every
figure is generated from the *same* source of truth (20 seeds + ablations).

Exposes:
    load_main()      -> dict with per-seed means + pooled eval episodes
    load_ablations() -> dict {label: {seeds, ct_mean, ...}}
    load_curves()    -> per-seed training curves for LLM-HRL / PureDRL
────────────────────────────────────────────────────────────────────────────
"""
import os, glob, re
import numpy as np
import pandas as pd

OUTPUT = r'E:\SYT\SID_1031\SID_LLM_1031\iplt\output'
SUPP_CSV = r'E:\SYT\SID_1031\SID_LLM_1031\iplt\paper_hicss\supplementary_results_summary.csv'
FAIL_THRESH = 3500          # seconds — catastrophic-failure threshold (tau)
CT_COL = 'AVG_ULOT_CYCLE_TIME'

# HICSS 5 seeds (original) + expansion 15 seeds = 20 total
HICSS_SEEDS = [42, 123, 777, 2024, 7]
EXP_SEEDS   = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
               1001, 1002, 1003, 1004, 1005]

# Hard-coded HICSS per-seed eval means (folders not all present under output/)
HICSS_DRL = {42: 4430, 123: 2820, 777: 2818, 2024: 4554, 7: 4474}
HICSS_LLM = {42: 3126, 123: 2806, 777: 3309, 2024: 3298, 7: 2815}

_SKIP = ['u20', 'u12', 'u4', 'u6', 'u8', 'u2', 'n10', 'rep', 'sanity',
         'sparse', 'nostate', 'mtx', 'abl', 'Ablation']


def _eval_df(folder):
    csvs = glob.glob(os.path.join(OUTPUT, folder, 'csv', '*.csv'))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    ev = df[df['is_eval'] == True]
    return ev if len(ev) >= 10 else None


def _train_series(folder):
    csvs = glob.glob(os.path.join(OUTPUT, folder, 'csv', '*.csv'))
    if not csvs:
        return None
    df = pd.read_csv(csvs[0])
    tr = df[df['is_eval'] == False]
    return tr[CT_COL].values if len(tr) else None


# ──────────────────────────────────────────────────────────────────────────
def load_main():
    """Return per-seed means and pooled eval-episode CT for both conditions."""
    drl_mean = dict(HICSS_DRL)
    llm_mean = dict(HICSS_LLM)
    drl_pool, llm_pool = [], []          # pooled per-episode eval CT
    drl_seed_eps, llm_seed_eps = {}, {}  # per-seed list of eval-episode CT

    for folder in sorted(os.listdir(OUTPUT)):
        if any(k in folder for k in _SKIP):
            continue
        if not any(folder.endswith('_s' + str(s)) for s in EXP_SEEDS):
            continue
        ev = _eval_df(folder)
        if ev is None:
            continue
        seed = int(ev['seed'].iloc[0])
        vals = ev[CT_COL].values
        if 'PureDRL' in folder:
            drl_mean[seed] = float(np.mean(vals))
            drl_pool.extend(vals); drl_seed_eps[seed] = vals
        elif 'FullLLM' in folder:
            llm_mean[seed] = float(np.mean(vals))
            llm_pool.extend(vals); llm_seed_eps[seed] = vals

    return {
        'drl_mean': drl_mean, 'llm_mean': llm_mean,
        'drl_pool': np.array(drl_pool), 'llm_pool': np.array(llm_pool),
        'drl_seed_eps': drl_seed_eps, 'llm_seed_eps': llm_seed_eps,
    }


# ──────────────────────────────────────────────────────────────────────────
ABL_MAP = {
    'Ablation_SoftReward':          'Soft Reward',
    'Ablation_RandomRules':         'Random Rules',
    'Ablation_StaticRules_Every5':  'Static (5 ep)',
    'Ablation_StaticRules_Fixed':   'Static (fixed)',
    'Ablation_StaticRules_Every10': 'Static (10 ep)',
    'Ablation_RandomMask':          'Random Mask',
}


def load_ablations():
    out = {v: {} for v in ABL_MAP.values()}
    for folder in sorted(os.listdir(OUTPUT)):
        for kw, lbl in ABL_MAP.items():
            if re.search(re.escape(kw) + r'_s\d+', folder):
                ev = _eval_df(folder)
                if ev is None:
                    continue
                seed = int(ev['seed'].iloc[0])
                out[lbl][seed] = float(ev[CT_COL].mean())
    # collapse to arrays
    res = {}
    for lbl, d in out.items():
        cts = np.array(list(d.values()))
        if len(cts) == 0:
            continue
        res[lbl] = {
            'cts': cts,
            'mean': float(cts.mean()),
            'std': float(cts.std(ddof=1)) if len(cts) > 1 else 0.0,
            'fail': int((cts > FAIL_THRESH).sum()),
            'n': len(cts),
            'worst': float(cts.max()),
        }
    return res


# ──────────────────────────────────────────────────────────────────────────
def load_rulebased():
    """Rule-Based ablation (fixed hand-crafted rule, no LLM adaptation),
    loaded from the HICSS supplementary CSV (5 seeds)."""
    df = pd.read_csv(SUPP_CSV)
    rb = df[df['condition'] == 'RuleBased']
    cts = rb['eval_CT_mean_s'].values.astype(float)
    return {
        'cts': cts,
        'mean': float(cts.mean()),
        'std': float(cts.std(ddof=1)),
        'fail': int((cts > FAIL_THRESH).sum()),
        'n': len(cts),
        'worst': float(cts.max()),
    }


def _is_curve_folder(folder):
    """True if folder is a main-experiment run (expansion seeds OR HICSS u20 seeds)."""
    # 15 expansion seeds: no skip keywords, ends with _s{seed}
    if not any(k in folder for k in _SKIP):
        if any(folder.endswith('_s' + str(s)) for s in EXP_SEEDS):
            return True
    # 5 original HICSS seeds: u20_mtx_e100_s{seed}
    if 'u20_mtx_e100' in folder:
        if any(folder.endswith('_s' + str(s)) for s in HICSS_SEEDS):
            return True
    return False


def load_curves():
    """Per-seed training AND evaluation curves (LLM-HRL / PureDRL),
    all 20 seeds. Each run = 100 episodes: train ep1-70 (is_eval=False),
    eval ep71-100 (is_eval=True). Returns training arrays plus per-seed eval means.
    """
    drl, llm = [], []            # training series (ep 1-70)
    drl_ev, llm_ev = [], []      # per-seed eval-episode arrays (ep 71-100)
    for folder in sorted(os.listdir(OUTPUT)):
        if not _is_curve_folder(folder):
            continue
        csvs = glob.glob(os.path.join(OUTPUT, folder, 'csv', '*.csv'))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        tr = df.loc[~df['is_eval'], CT_COL].values
        ev = df.loc[df['is_eval'], CT_COL].values
        if len(tr) == 0 or len(ev) == 0:
            continue
        if 'PureDRL' in folder:
            drl.append(tr); drl_ev.append(ev)
        elif 'FullLLM' in folder:
            llm.append(tr); llm_ev.append(ev)
    return {'drl': drl, 'llm': llm, 'drl_ev': drl_ev, 'llm_ev': llm_ev}


if __name__ == '__main__':
    m = load_main()
    dn = np.array(list(m['drl_mean'].values()))
    ln = np.array(list(m['llm_mean'].values()))
    print(f"PureDRL : n={len(dn)}  mean={dn.mean():.0f}  fail={(dn>FAIL_THRESH).sum()}")
    print(f"LLM-HRL : n={len(ln)}  mean={ln.mean():.0f}  fail={(ln>FAIL_THRESH).sum()}")
    print(f"pooled eval episodes: DRL={len(m['drl_pool'])}  LLM={len(m['llm_pool'])}")
    print("\nAblations:")
    for lbl, d in load_ablations().items():
        print(f"  {lbl:<16} n={d['n']} mean={d['mean']:.0f} fail={d['fail']}/{d['n']}")
    c = load_curves()
    print(f"\ncurves: DRL={len(c['drl'])} seeds, LLM={len(c['llm'])} seeds")
