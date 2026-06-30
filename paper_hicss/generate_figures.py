"""
generate_figures.py
────────────────────────────────────────────────────────────────────────────────
Publication-quality figures for HICSS-60 paper.
Generates:
  fig_sparsity_sweep.pdf/png   — sweep CT + failure rate
  fig_nondeterminism.pdf/png   — two-run divergence under same seed
  fig_mechanism.pdf/png        — (a) reward fraction + (b) convergence
────────────────────────────────────────────────────────────────────────────────
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_BASE = r'E:\SYT\SID_1031\SID_LLM_1031\iplt\output'
DPI = 300

# ── Global style (Times-like, publication-ready) ──────────────────────────────
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size':         8,
    'axes.labelsize':    8,
    'axes.titlesize':    9,
    'xtick.labelsize':   7.5,
    'ytick.labelsize':   7.5,
    'legend.fontsize':   7.5,
    'legend.framealpha': 0.9,
    'figure.dpi':        DPI,
    'savefig.dpi':       DPI,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.04,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'lines.linewidth':   1.4,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# ── Colour palette (colour-blind friendly) ────────────────────────────────────
C_DRL  = '#2166AC'   # blue
C_LLM  = '#D6604D'   # red-orange
C_FAIL = '#F4A582'   # light salmon (failure bar)
C_GRAY = '#808080'
C_EVAL_BG = '#F0F4F8'

FAIL_THRESH = 3_500   # seconds


# ══════════════════════════════════════════════════════════════════════════════
# Helper: load CSV for a given condition / urgent count / seed
# ══════════════════════════════════════════════════════════════════════════════
def load_run(cond_str: str, u: int, seed: int):
    """Return DataFrame for matching folder, or None."""
    matches = [
        f for f in os.listdir(DATA_BASE)
        if f'u{u}_mtx_e100_s{seed}' in f and cond_str in f
    ]
    if not matches:
        return None
    csv_files = glob.glob(os.path.join(DATA_BASE, matches[0], 'csv', 'results_*.csv'))
    return pd.read_csv(csv_files[0]) if csv_files else None


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Sparsity Sweep
# ══════════════════════════════════════════════════════════════════════════════
def fig_sparsity_sweep():
    # ── Data ──────────────────────────────────────────────────────────────────
    # u4: earlier-protocol runs (consistent, N=6)
    u4_drl  = [2281, 2281, 2296, 2296, 2274, 2274]
    u4_llm  = [2274, 2274, 2274, 2278, 2281, 2281]
    # u12, u20: mtx-protocol (N=5)
    u12_drl  = [4590, 5608, 2554, 2554, 4549]
    u12_llm  = [4593, 4595, 3025, 2554, 2557]
    u20_drl  = [4430, 2820, 2818, 4554, 4474]
    u20_llm  = [3126, 2806, 3309, 3298, 2815]

    groups = {
        '4\n(1.6%)':  (u4_drl,  u4_llm),
        '12\n(4.7%)': (u12_drl, u12_llm),
        '20\n(7.7%)': (u20_drl, u20_llm),
    }

    means_drl, means_llm = [], []
    stds_drl,  stds_llm  = [], []
    fail_drl,  fail_llm  = [], []
    labels = list(groups.keys())

    for key, (d, l) in groups.items():
        means_drl.append(np.mean(d)); stds_drl.append(np.std(d, ddof=1))
        means_llm.append(np.mean(l)); stds_llm.append(np.std(l, ddof=1))
        fail_drl.append(sum(v > FAIL_THRESH for v in d) / len(d) * 100)
        fail_llm.append(sum(v > FAIL_THRESH for v in l) / len(l) * 100)

    x = np.arange(len(labels))
    w = 0.32

    fig, ax1 = plt.subplots(figsize=(3.5, 2.8))
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)

    # CT bars
    bars_d = ax1.bar(x - w/2, means_drl, w, yerr=stds_drl,
                     color=C_DRL, alpha=0.85, capsize=3,
                     error_kw={'elinewidth': 0.9, 'ecolor': '#333'},
                     label='Pure DRL', zorder=3)
    bars_l = ax1.bar(x + w/2, means_llm, w, yerr=stds_llm,
                     color=C_LLM, alpha=0.85, capsize=3,
                     error_kw={'elinewidth': 0.9, 'ecolor': '#333'},
                     label='LLM-HRL (ours)', zorder=3)

    # Failure rate line (on ax2)
    ax2.plot(x - w/2, fail_drl, 'o--', color=C_DRL,
             markersize=5, linewidth=1.2, alpha=0.7, zorder=4)
    ax2.plot(x + w/2, fail_llm, 's--', color=C_LLM,
             markersize=5, linewidth=1.2, alpha=0.7, zorder=4)

    # Annotate fail rates — stagger DRL/LLM when both non-zero to avoid overlap
    for xi, (fd, fl) in enumerate(zip(fail_drl, fail_llm)):
        both_nonzero = (fd > 0 and fl > 0)
        off_d = (8,  0) if both_nonzero else (0, 5)
        off_l = (-8, 0) if both_nonzero else (0, 5)

        label_d = f'{fd:.0f}%' if fd > 0 else '0%'
        fw_d = 'bold' if fd > 0 else 'normal'
        ax2.annotate(label_d, (xi - w/2, fd),
                     textcoords='offset points', xytext=off_d,
                     ha='center', fontsize=6.5, color=C_DRL, fontweight=fw_d)

        label_l = f'{fl:.0f}%' if fl > 0 else '0%'
        fw_l = 'bold' if fl > 0 else 'normal'
        ax2.annotate(label_l, (xi + w/2, fl),
                     textcoords='offset points', xytext=off_l,
                     ha='center', fontsize=6.5, color=C_LLM, fontweight=fw_l)

    # Threshold line
    ax1.axhline(FAIL_THRESH, color='#555', linestyle=':', linewidth=0.9, zorder=2)
    ax1.text(2.48, FAIL_THRESH + 60, 'Failure\nthreshold', ha='right',
             fontsize=6, color='#555', va='bottom')

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Urgent lots: {lb}' for lb in ['4\n(1.6%)', '12\n(4.7%)', '20\n(7.7%)']])
    ax1.set_xlabel('Number of urgent lots (% of total)')
    ax1.set_ylabel('Urgent-Lot Cycle Time (s)')
    ax2.set_ylabel('Catastrophic Failure Rate (%)')
    ax2.set_ylim(-5, 90)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))

    ax1.set_ylim(0, 6400)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.1f}k' if v >= 1000 else f'{v:.0f}'))

    # Legend
    patch_d = mpatches.Patch(color=C_DRL, alpha=0.85, label='Pure DRL — CT (bar) / Fail rate (●)')
    patch_l = mpatches.Patch(color=C_LLM, alpha=0.85, label='LLM-HRL  — CT (bar) / Fail rate (■)')
    ax1.legend(handles=[patch_d, patch_l], loc='upper left', fontsize=6.5,
               framealpha=0.9, edgecolor='#ccc')

    ax1.set_title('Effect of Urgent-Lot Count on\nCycle Time and Failure Rate', pad=6)
    ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    _save(fig, 'fig_sparsity_sweep')
    print('  OK fig_sparsity_sweep')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Nondeterminism
# ══════════════════════════════════════════════════════════════════════════════
def fig_nondeterminism():
    # Two PureDRL u4-protocol runs with same seed (s42)
    run_a = [2919,5255,4031,3142,2252,2252,6034,3587,4477,2252,
             3364,4588,2697,2474,2252,2252,3142,2585,3809,2363,
             2252,2363,2252,5811,2252,4588,4811,2252,2140,2252,
             2585,3921,2252,2474,2252,2252,2252,2363,2252,2252,
             2252,2252,2252,2252,2252,2252,2474,2252,2252,2252]
    run_b = [2920,5255,4031,3364,2919,3921,6034,3364,4477,2252,
             4143,4477,3142,4143,3364,2252,2252,3809,4588,3031,
             2252,2363,2252,5366,6035,4588,4811,2252,5590,2252,
             3142,2252,3809,2474,2697,3921,3253,3698,3587,3476,
             3921,3699,3699,3921,3809,3587,3921,3810,3587,3699]

    episodes = np.arange(1, 51)
    EVAL_START = 36   # episodes 36-50 are eval

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    # Eval background
    ax.axvspan(EVAL_START - 0.5, 50.5, color=C_EVAL_BG, zorder=0,
               label='Evaluation region')

    # Smooth with rolling avg (window=4)
    def smooth(arr, w=4):
        return pd.Series(arr).rolling(w, min_periods=1, center=True).mean().values

    ax.plot(episodes, run_a, color=C_DRL, alpha=0.25, linewidth=0.7, zorder=2)
    ax.plot(episodes, run_b, color=C_LLM, alpha=0.25, linewidth=0.7, zorder=2)
    ax.plot(episodes, smooth(run_a), color=C_DRL, linewidth=1.6, zorder=3,
            label=f'Run A  (eval CT = 2,274 s  [PASS])')
    ax.plot(episodes, smooth(run_b), color=C_LLM, linewidth=1.6, zorder=3,
            label=f'Run B  (eval CT = 3,706 s  [FAIL])')

    # Divergence arrow annotation
    ax.annotate('', xy=(22, 5250), xytext=(22, 3900),
                arrowprops=dict(arrowstyle='<->', color='#444', lw=0.9))
    ax.text(22.5, 4570, 'divergence\nbegins', fontsize=6, color='#444', va='center')

    # Eval label
    ax.text((EVAL_START + 50) / 2, 6300, 'Evaluation\n(frozen policy)',
            ha='center', va='top', fontsize=6.5, color='#555')

    # Threshold line
    ax.axhline(FAIL_THRESH, color='#555', linestyle=':', linewidth=0.9, zorder=2)
    ax.text(1, FAIL_THRESH + 120, f'Failure threshold ({FAIL_THRESH:,} s)',
            fontsize=6, color='#555')

    ax.set_xlim(0.5, 50.5)
    ax.set_ylim(1200, 7000)
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Urgent-Lot Cycle Time (s)')
    ax.set_title('Digital-Twin Nondeterminism:\nIdentical Seed, Divergent Outcomes', pad=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v/1000:.1f}k'))
    ax.legend(loc='upper right', fontsize=6.5, framealpha=0.9, edgecolor='#ccc')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    _save(fig, 'fig_nondeterminism')
    print('  OK fig_nondeterminism')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Sparse Reward Mechanism (two panels)
# ══════════════════════════════════════════════════════════════════════════════
def fig_mechanism():
    # ── u20 convergence (all 5 seeds) ─────────────────────────────────────────
    seeds = [42, 123, 777, 2024, 7]
    drl_trains, llm_trains = [], []
    drl_eval_means, llm_eval_means = [], []

    for s in seeds:
        df_d = load_run('PureDRL',          20, s)
        df_l = load_run('Proposed_FullLLM', 20, s)
        if df_d is None or df_l is None:
            continue
        col = 'AVG_ULOT_CYCLE_TIME'
        drl_trains.append(df_d.loc[~df_d['is_eval'], col].values)
        llm_trains.append(df_l.loc[~df_l['is_eval'], col].values)
        drl_eval_means.append(df_d.loc[df_d['is_eval'], col].mean())
        llm_eval_means.append(df_l.loc[df_l['is_eval'], col].mean())

    drl_arr  = np.array(drl_trains)
    llm_arr  = np.array(llm_trains)
    ep_train = np.arange(1, drl_arr.shape[1] + 1)

    def smooth_arr(arr, w=7):
        return np.array([pd.Series(row).rolling(w, min_periods=1, center=True).mean().values
                         for row in arr])

    drl_sm   = smooth_arr(drl_arr)
    llm_sm   = smooth_arr(llm_arr)
    drl_mean = drl_sm.mean(axis=0);  drl_std = drl_sm.std(axis=0, ddof=1)
    llm_mean = llm_sm.mean(axis=0);  llm_std = llm_sm.std(axis=0, ddof=1)

    # ── Single-panel layout ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.6, 2.9))

    # Individual seed thin lines
    for d_row, l_row in zip(drl_arr, llm_arr):
        ax.plot(ep_train, pd.Series(d_row).rolling(7, min_periods=1, center=True).mean(),
                color=C_DRL, alpha=0.15, linewidth=0.7, zorder=2)
        ax.plot(ep_train, pd.Series(l_row).rolling(7, min_periods=1, center=True).mean(),
                color=C_LLM, alpha=0.15, linewidth=0.7, zorder=2)

    # Mean ± std band
    ax.fill_between(ep_train, drl_mean - drl_std, drl_mean + drl_std,
                    color=C_DRL, alpha=0.12, zorder=2)
    ax.fill_between(ep_train, llm_mean - llm_std, llm_mean + llm_std,
                    color=C_LLM, alpha=0.12, zorder=2)
    ax.plot(ep_train, drl_mean, color=C_DRL, linewidth=2.0, zorder=4, label='Pure DRL')
    ax.plot(ep_train, llm_mean, color=C_LLM, linewidth=2.0, zorder=4, label='LLM-HRL (ours)')

    # Eval stats
    eval_drl_m = np.mean(drl_eval_means);  eval_drl_s = np.std(drl_eval_means, ddof=1)
    eval_llm_m = np.mean(llm_eval_means);  eval_llm_s = np.std(llm_eval_means, ddof=1)

    # Eval region shading + divider
    ax.axvspan(70.5, 100.5, color=C_EVAL_BG, zorder=1)
    ax.axvline(70.5, color='#888', linestyle='--', linewidth=0.9, zorder=3)
    ax.text(85, 6050, 'Evaluation\n(30 eps)', fontsize=6.5, color='#555',
            va='top', ha='center')

    # Eval mean bands
    ax.fill_betweenx([eval_drl_m - eval_drl_s, eval_drl_m + eval_drl_s],
                     70.5, 100.5, color=C_DRL, alpha=0.15, zorder=2)
    ax.plot([70.5, 100.5], [eval_drl_m, eval_drl_m],
            color=C_DRL, linewidth=2.0, zorder=4)
    ax.fill_betweenx([eval_llm_m - eval_llm_s, eval_llm_m + eval_llm_s],
                     70.5, 100.5, color=C_LLM, alpha=0.15, zorder=2)
    ax.plot([70.5, 100.5], [eval_llm_m, eval_llm_m],
            color=C_LLM, linewidth=2.0, zorder=4)

    # Result summary boxes
    ax.text(99, eval_drl_m + 200,
            f'DRL: {eval_drl_m:.0f}s\n(±{eval_drl_s:.0f})',
            fontsize=6.5, color=C_DRL, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_DRL,
                      alpha=0.9, lw=0.7), zorder=6)
    ax.text(99, eval_llm_m - 200,
            f'LLM: {eval_llm_m:.0f}s\n(±{eval_llm_s:.0f})',
            fontsize=6.5, color=C_LLM, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_LLM,
                      alpha=0.9, lw=0.7), zorder=6)

    # Failure threshold
    ax.axhline(FAIL_THRESH, color='#777', linestyle=':', linewidth=1.0, zorder=3)
    ax.text(1, FAIL_THRESH + 100, 'Failure threshold',
            fontsize=6, color='#666', ha='left', va='bottom')

    ax.set_xlim(0, 102)
    ax.set_ylim(1500, 6500)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Urgent-Lot Cycle Time (s)')
    ax.set_title('Training Convergence at 20 Urgent Lots\n'
                 r'(5 seeds; shaded = $\pm$1 SD)', pad=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v/1000:.1f}k'))
    ax.legend(loc='lower left', fontsize=7, framealpha=0.95, edgecolor='#ccc')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    _save(fig, 'fig_mechanism')
    print('  OK fig_mechanism')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Per-seed reliability detail (Table as heatmap-style)
# ══════════════════════════════════════════════════════════════════════════════
def fig_perseed():
    seeds   = [42, 123, 777, 2024, 7]
    drl_ct  = [4430, 2820, 2818, 4554, 4474]
    llm_ct  = [3126, 2806, 3309, 3298, 2815]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.set_aspect('auto')

    x = np.arange(len(seeds))
    w = 0.34

    bars_d = ax.bar(x - w/2, drl_ct, w, color=C_DRL, alpha=0.85, label='Pure DRL', zorder=3)
    bars_l = ax.bar(x + w/2, llm_ct, w, color=C_LLM, alpha=0.85, label='LLM-HRL (ours)', zorder=3)

    # Failure threshold
    ax.axhline(FAIL_THRESH, color='#444', linestyle='--', linewidth=1.0, zorder=2)
    ax.text(len(seeds) - 0.5 + w/2 + 0.08, FAIL_THRESH + 50,
            'Failure\nthreshold', fontsize=6, color='#444', va='bottom', ha='left')

    # Shade failure zone
    ax.axhspan(FAIL_THRESH, 5800, color='#FFF0EE', zorder=0)

    # Value labels on bars — alternate offset when DRL and LLM values are close
    for i, (bar_d, bar_l, dv, lv) in enumerate(zip(bars_d, bars_l, drl_ct, llm_ct)):
        is_fail = dv > FAIL_THRESH
        close   = abs(dv - lv) < 150   # values nearly equal → stagger vertically
        off_d = 180 if close else 55
        off_l = -200 if close else 55   # LLM label goes below bar top when close
        color_d = '#B22222' if is_fail else C_DRL
        ax.text(bar_d.get_x() + bar_d.get_width()/2, dv + off_d,
                f'{dv:,}', ha='center', va='bottom', fontsize=6.2,
                color=color_d, fontweight='bold' if is_fail else 'normal')
        ax.text(bar_l.get_x() + bar_l.get_width()/2,
                lv + (off_l if close else 55),
                f'{lv:,}', ha='center',
                va='top' if close else 'bottom',
                fontsize=6.2, color=C_LLM)

    # Delta arrows for failed DRL seeds
    for i, (d, l) in enumerate(zip(drl_ct, llm_ct)):
        if d > FAIL_THRESH:
            delta = d - l
            mid_x = x[i]
            ax.annotate('', xy=(mid_x + w/2, l + 120), xytext=(mid_x - w/2, d - 120),
                        arrowprops=dict(arrowstyle='->', color='#555',
                                        lw=0.9, connectionstyle='arc3,rad=0'))
            ax.text(mid_x + 0.01, (d + l) / 2, f'−{delta:,}s',
                    ha='left', va='center', fontsize=5.8, color='#333',
                    style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels([f'seed\n{s}' for s in seeds])
    ax.set_ylabel('Eval. Urgent-Lot Cycle Time (s)')
    ax.set_ylim(1800, 5800)
    ax.set_title('Per-Seed Results at 20 Urgent Lots\n(shaded = catastrophic failure zone)',
                 pad=5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f'{v/1000:.1f}k'))
    ax.legend(loc='upper center', ncol=2, fontsize=7, framealpha=0.9, edgecolor='#ccc')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    _save(fig, 'fig_perseed')
    print('  OK fig_perseed')


# ══════════════════════════════════════════════════════════════════════════════
# Save helper
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, name: str):
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating publication figures...')
    fig_sparsity_sweep()
    fig_nondeterminism()
    fig_mechanism()
    fig_perseed()
    print('\nAll figures saved to:', OUT_DIR)
    print('Files: fig_sparsity_sweep, fig_nondeterminism, fig_mechanism, fig_perseed  (.pdf / .png)')
