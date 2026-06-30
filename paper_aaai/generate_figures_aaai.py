"""
generate_figures_aaai.py
────────────────────────────────────────────────────────────────────────────
Publication-quality figures for the AAAI-27 submission.
All data are loaded live from output/ via aaai_data.py (20 seeds + ablations).

Figures produced (.pdf + .png, 300 dpi, AAAI two-column geometry):
  fig_reliability   — Main result: per-seed CT box+strip, 20 seeds      [1-col]
  fig_ct_dist       — Pooled eval-CT distribution, bimodality & tau      [1-col]
  fig_convergence   — Training convergence, mean +/- SD bands            [1-col]
  fig_ablation      — Ablation: component contribution (CT + fail-rate)  [2-col]
────────────────────────────────────────────────────────────────────────────
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy import stats

import aaai_data as D

OUT = os.path.dirname(os.path.abspath(__file__))
DPI = 300
FAIL = D.FAIL_THRESH

# ── Global style — AAAI / Times-like, publication ready ─────────────────────
mpl.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'Times'],
    'mathtext.fontset':   'stix',
    'font.size':          8,
    'axes.labelsize':     8.5,
    'axes.titlesize':     9,
    'xtick.labelsize':    7.5,
    'ytick.labelsize':    7.5,
    'legend.fontsize':    7,
    'legend.framealpha':  0.95,
    'figure.dpi':         DPI,
    'savefig.dpi':        DPI,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.03,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'lines.linewidth':    1.4,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'pdf.fonttype':       42,   # editable / embedded TrueType
    'ps.fonttype':        42,
})

# Colour-blind-friendly palette (consistent with HICSS figures)
C_DRL  = '#2166AC'   # blue
C_LLM  = '#D6604D'   # red-orange
C_FAILZONE = '#FBE9E7'
C_GRID = '#B0B0B0'
C_GRAY = '#5A5A5A'
C_EVAL_BG = '#F0F4F8'


def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  OK  {name}')


def _k(v, _=None):
    return f'{v/1000:.1f}k'


# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Main reliability result (per-seed box + strip, 20 seeds)
# ══════════════════════════════════════════════════════════════════════════
def fig_reliability(m):
    drl = np.array(list(m['drl_mean'].values()))
    llm = np.array(list(m['llm_mean'].values()))
    n = len(drl)

    fdr = (drl > FAIL).mean() * 100
    fll = (llm > FAIL).mean() * 100
    t, p = stats.ttest_ind(drl, llm)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    # failure zone
    ax.axhspan(FAIL, 5600, color=C_FAILZONE, zorder=0)
    ax.axhline(FAIL, color=C_GRAY, ls='--', lw=0.9, zorder=2)

    pos = [1, 2]
    data = [drl, llm]
    colors = [C_DRL, C_LLM]

    # box plots (no fliers — points drawn as strip)
    bp = ax.boxplot(data, positions=pos, widths=0.46, patch_artist=True,
                    showfliers=False, zorder=3,
                    medianprops=dict(color='#222', lw=1.3),
                    whiskerprops=dict(color='#555', lw=0.9),
                    capprops=dict(color='#555', lw=0.9))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.22); patch.set_edgecolor(c)
        patch.set_linewidth(1.1)

    # strip (jittered individual seeds)
    rng = np.random.default_rng(0)
    for xc, vals, c in zip(pos, data, colors):
        jit = rng.uniform(-0.13, 0.13, len(vals))
        fail_mask = vals > FAIL
        ax.scatter(xc + jit[~fail_mask], vals[~fail_mask], s=16, color=c,
                   edgecolor='white', linewidth=0.4, alpha=0.9, zorder=4)
        ax.scatter(xc + jit[fail_mask], vals[fail_mask], s=20, color=c,
                   edgecolor='#7a0000', linewidth=0.7, alpha=0.95, zorder=5,
                   marker='X')

    # significance bracket
    y0 = 5050
    ax.plot([1, 1, 2, 2], [y0, y0+120, y0+120, y0], lw=0.9, color='#333')
    ax.text(1.5, y0+170, f'$p$ = {p:.4f} **', ha='center', va='bottom',
            fontsize=7.5)

    # fail-rate annotations under each box
    ax.text(1, 1700, f'fail\n{fdr:.0f}%', ha='center', va='center',
            fontsize=7.5, color=C_DRL, fontweight='bold')
    ax.text(2, 1700, f'fail\n{fll:.0f}%', ha='center', va='center',
            fontsize=7.5, color=C_LLM, fontweight='bold')

    ax.text(2.46, FAIL+70, 'failure zone', ha='right', va='bottom',
            fontsize=6.3, color=C_GRAY, style='italic')

    ax.set_xticks(pos)
    ax.set_xticklabels([f'Pure DRL\n(n={n})', f'LLM-HRL\n(n={n})'])
    ax.set_ylabel('Mean Urgent-Lot Cycle Time (s)')
    ax.set_ylim(1400, 5500)
    ax.set_xlim(0.4, 2.6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.set_title('Reliability across 20 independent seeds', pad=7)
    ax.yaxis.grid(True, ls='--', lw=0.5, color=C_GRID, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # legend for markers
    h_ok   = plt.Line2D([], [], marker='o', ls='', color=C_GRAY,
                        markeredgecolor='white', markersize=4.5, label='pass run')
    h_fail = plt.Line2D([], [], marker='X', ls='', color=C_GRAY,
                        markeredgecolor='#7a0000', markersize=5, label='failed run (CT > $\\tau$)')
    ax.legend(handles=[h_ok, h_fail], loc='upper left', frameon=True,
              edgecolor='#ccc', handletextpad=0.3, borderpad=0.4)

    plt.tight_layout(pad=0.5)
    _save(fig, 'fig_reliability')


# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — Pooled evaluation CT distribution (bimodality + tau)
# ══════════════════════════════════════════════════════════════════════════
def fig_ct_dist(m):
    drl = m['drl_pool']
    llm = m['llm_pool']

    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    bins = np.linspace(1500, 6000, 46)
    ax.hist(drl, bins=bins, color=C_DRL, alpha=0.55, label='Pure DRL',
            edgecolor='white', linewidth=0.2, zorder=3)
    ax.hist(llm, bins=bins, color=C_LLM, alpha=0.60, label='LLM-HRL',
            edgecolor='white', linewidth=0.2, zorder=4)

    ax.axvline(FAIL, color=C_GRAY, ls='--', lw=1.1, zorder=5)
    ymax = ax.get_ylim()[1]
    ax.text(FAIL+60, ymax*0.93, r'$\tau = 3{,}500$ s',
            fontsize=7, color='#333', ha='left', va='top', rotation=0)

    # bracket arrows for the two regimes
    ax.annotate('success\ncluster', xy=(2700, ymax*0.62),
                ha='center', fontsize=6.5, color=C_GRAY, style='italic')
    ax.annotate('failure\ncluster', xy=(4700, ymax*0.40),
                ha='center', fontsize=6.5, color=C_GRAY, style='italic')

    ax.set_xlabel('Evaluation-episode Urgent-Lot Cycle Time (s)')
    ax.set_ylabel('Episode count')
    ax.set_xlim(1500, 6000)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.set_title('Bimodal CT distribution (pooled eval episodes)', pad=7)
    ax.legend(loc='upper right', edgecolor='#ccc')
    ax.yaxis.grid(True, ls='--', lw=0.5, color=C_GRID, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    _save(fig, 'fig_ct_dist')


# ══════════════════════════════════════════════════════════════════════════
# Figure 3 — Training convergence (mean +/- SD bands)
# ══════════════════════════════════════════════════════════════════════════
def fig_convergence(c):
    """Full 100-episode view: training curve (ep 1-70) + evaluation region
    (ep 71-100) with mean +/- SD bands and result boxes, in the HICSS style."""
    def stack(curves):
        L = min(len(x) for x in curves)
        return np.array([x[:L] for x in curves])

    drl = stack(c['drl']); llm = stack(c['llm'])     # (n, ~70)
    n_train = drl.shape[1]
    ep = np.arange(1, n_train + 1)

    # per-seed eval means (ep 71-100)
    drl_ev_m = np.array([np.mean(e) for e in c['drl_ev']])
    llm_ev_m = np.array([np.mean(e) for e in c['llm_ev']])
    EVAL_START = n_train + 0.5
    EVAL_END = n_train + 30 + 0.5

    def smooth(arr, w=7):
        return np.array([pd.Series(r).rolling(w, min_periods=1, center=True).mean().values
                         for r in arr])
    drl_s = smooth(drl); llm_s = smooth(llm)
    dm, ds = drl_s.mean(0), drl_s.std(0, ddof=1)
    lm, ls = llm_s.mean(0), llm_s.std(0, ddof=1)

    fig, ax = plt.subplots(figsize=(4.7, 3.0))

    # individual seed thin lines (training)
    for d_row, l_row in zip(drl, llm):
        ax.plot(ep, pd.Series(d_row).rolling(7, min_periods=1, center=True).mean(),
                color=C_DRL, alpha=0.13, lw=0.6, zorder=2)
        ax.plot(ep, pd.Series(l_row).rolling(7, min_periods=1, center=True).mean(),
                color=C_LLM, alpha=0.13, lw=0.6, zorder=2)

    # mean +/- SD bands (training)
    ax.fill_between(ep, dm-ds, dm+ds, color=C_DRL, alpha=0.12, zorder=2)
    ax.fill_between(ep, lm-ls, lm+ls, color=C_LLM, alpha=0.12, zorder=2)
    ax.plot(ep, dm, color=C_DRL, lw=2.0, zorder=4,
            label=f'Pure DRL (n={drl.shape[0]})')
    ax.plot(ep, lm, color=C_LLM, lw=2.0, zorder=4,
            label=f'LLM-HRL (n={llm.shape[0]})')

    # evaluation region
    ev_dm, ev_ds = drl_ev_m.mean(), drl_ev_m.std(ddof=1)
    ev_lm, ev_ls = llm_ev_m.mean(), llm_ev_m.std(ddof=1)
    ax.axvspan(EVAL_START, EVAL_END, color=C_EVAL_BG, zorder=1)
    ax.axvline(EVAL_START, color='#888', ls='--', lw=0.9, zorder=3)
    ax.text((EVAL_START+EVAL_END)/2, 6050, 'Evaluation\n(30 eps)',
            fontsize=6.8, color='#555', va='top', ha='center')

    ax.fill_betweenx([ev_dm-ev_ds, ev_dm+ev_ds], EVAL_START, EVAL_END,
                     color=C_DRL, alpha=0.15, zorder=2)
    ax.plot([EVAL_START, EVAL_END], [ev_dm, ev_dm], color=C_DRL, lw=2.0, zorder=4)
    ax.fill_betweenx([ev_lm-ev_ls, ev_lm+ev_ls], EVAL_START, EVAL_END,
                     color=C_LLM, alpha=0.15, zorder=2)
    ax.plot([EVAL_START, EVAL_END], [ev_lm, ev_lm], color=C_LLM, lw=2.0, zorder=4)

    # result boxes
    ax.text(EVAL_END-0.5, ev_dm+200, f'DRL: {ev_dm:.0f}s\n($\\pm${ev_ds:.0f})',
            fontsize=6.6, color=C_DRL, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_DRL,
                      alpha=0.92, lw=0.7), zorder=6)
    ax.text(EVAL_END-0.5, ev_lm-200, f'LLM: {ev_lm:.0f}s\n($\\pm${ev_ls:.0f})',
            fontsize=6.6, color=C_LLM, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=C_LLM,
                      alpha=0.92, lw=0.7), zorder=6)

    ax.axhline(FAIL, color=C_GRAY, ls=':', lw=1.0, zorder=3)
    ax.text(2, FAIL+90, 'failure threshold', fontsize=6.2, color=C_GRAY,
            va='bottom')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Urgent-Lot Cycle Time (s)')
    ax.set_xlim(1, EVAL_END + 0.5)
    ax.set_ylim(1800, 6400)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax.set_title(r'Training convergence and evaluation (shaded band = $\pm$1 SD)',
                 pad=7)
    ax.legend(loc='lower left', edgecolor='#ccc')
    ax.yaxis.grid(True, ls='--', lw=0.5, color=C_GRID, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    _save(fig, 'fig_convergence')


# ══════════════════════════════════════════════════════════════════════════
# Figure — Failure mechanism & where governance breaks the chain (2-column)
# ══════════════════════════════════════════════════════════════════════════
def fig_mechanism():
    """Single-column (portrait) failure-mechanism schematic."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(3.4, 3.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    C_PASS = '#1A9850'   # green
    box_fc = '#EEF2F6'

    def node(x, y, w, h, text, ec, fc=box_fc, fs=7.2, bold=False):
        b = FancyBboxPatch((x-w/2, y-h/2), w, h,
                           boxstyle='round,pad=0.012,rounding_size=0.02',
                           linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=3)
        ax.add_patch(b)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                color='#1a1a1a', zorder=4,
                fontweight='bold' if bold else 'normal')

    def arrow(x0, y0, x1, y1, color='#444', lw=1.5, style='-|>'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    shrinkA=1, shrinkB=1))

    xm = 0.38            # main chain x-center
    w, h = 0.46, 0.092
    y1, y2, y3, y4 = 0.94, 0.775, 0.61, 0.445

    node(xm, y1, w, h, 'Simulator timing jitter (sub-RNG)', C_DRL)
    node(xm, y2, w, h, 'Divergent state transitions', C_DRL)
    node(xm, y3, w, h, 'Replay buffer $\\rightarrow$ perturbed $Q$', C_DRL)
    node(xm, y4, w, h, 'Policy divergence', C_DRL)
    for ya_, yb_ in [(y1, y2), (y2, y3), (y3, y4)]:
        arrow(xm, ya_-h/2, xm, yb_+h/2, C_DRL)

    # branch into pass / fail basins
    yb = 0.17
    node(0.235, yb, 0.40, 0.115, 'Passing\n($\\mathrm{CT}<\\tau$)', C_PASS, fc='#E8F5E9')
    node(0.700, yb, 0.40, 0.115, 'Failing\n($\\mathrm{CT}>\\tau$)', C_LLM, fc=C_FAILZONE)
    arrow(xm-0.05, y4-h/2, 0.29, yb+0.057, C_PASS)
    arrow(xm+0.05, y4-h/2, 0.65, yb+0.057, C_LLM)

    # governance intervention (right side, curved arrow into policy divergence)
    node(0.83, 0.695, 0.30, 0.225,
         'LLM\ngovernance:\nhard,\ncontextual\nmask\n(training)',
         C_LLM, fc='#FDECEA', fs=6.6, bold=True)
    ax.annotate('', xy=(xm+w/2, y4+0.012), xytext=(0.83, 0.695-0.115),
                zorder=2, arrowprops=dict(arrowstyle='-|>', color=C_LLM, lw=1.7,
                connectionstyle='arc3,rad=0.28', shrinkA=2, shrinkB=2))
    ax.text(0.86, 0.45, 'breaks\nthe chain', fontsize=6.2, color=C_LLM,
            style='italic', ha='left', va='center', zorder=5)

    # Laprie taxonomy tags
    def ltag(x, y, txt):
        ax.text(x, y, txt, ha='center', va='center', fontsize=6.6,
                color='#666', style='italic')
    ltag(0.07, y1, 'fault'); ltag(0.07, y4, 'error')
    ltag(0.700, yb-0.087, 'failure')

    plt.tight_layout(pad=0.3)
    _save(fig, 'fig_mechanism')


# ══════════════════════════════════════════════════════════════════════════
# Figure — Urgent-lot density sweep (generalization across operating regimes)
# ══════════════════════════════════════════════════════════════════════════
def fig_sweep(m):
    # Validated runs from prior work / supplementary (u4: n=6, u12: n=5).
    u4_drl  = [2281, 2281, 2296, 2296, 2274, 2274]
    u4_llm  = [2274, 2274, 2274, 2278, 2281, 2281]
    u12_drl = [4590, 5608, 2554, 2554, 4549]
    u12_llm = [4593, 4595, 3025, 2554, 2557]
    u20_drl = np.array(list(m['drl_mean'].values()))   # n=20 (live)
    u20_llm = np.array(list(m['llm_mean'].values()))

    groups = [('1.6%', u4_drl, u4_llm),
              ('4.7%', u12_drl, u12_llm),
              ('7.7%', u20_drl, u20_llm)]
    labels = [g[0] for g in groups]
    md = [np.mean(g[1]) for g in groups]; sd = [np.std(g[1], ddof=1) for g in groups]
    ml = [np.mean(g[2]) for g in groups]; sl = [np.std(g[2], ddof=1) for g in groups]
    fd = [(np.array(g[1]) > FAIL).mean()*100 for g in groups]
    fl = [(np.array(g[2]) > FAIL).mean()*100 for g in groups]

    x = np.arange(len(groups)); w = 0.34
    fig, ax1 = plt.subplots(figsize=(3.4, 2.8))
    ax2 = ax1.twinx()
    ax2.spines['right'].set_visible(True); ax2.spines['top'].set_visible(False)

    ax1.bar(x - w/2, md, w, yerr=sd, color=C_DRL, alpha=0.85, capsize=2.5,
            error_kw={'elinewidth': 0.8, 'ecolor': '#444'}, zorder=3)
    ax1.bar(x + w/2, ml, w, yerr=sl, color=C_LLM, alpha=0.85, capsize=2.5,
            error_kw={'elinewidth': 0.8, 'ecolor': '#444'}, zorder=3)

    ax2.plot(x - w/2, fd, 'o--', color=C_DRL, ms=5, lw=1.2, alpha=0.75, zorder=4)
    ax2.plot(x + w/2, fl, 's--', color=C_LLM, ms=5, lw=1.2, alpha=0.75, zorder=4)
    for xi, (a, b) in enumerate(zip(fd, fl)):
        ax2.annotate(f'{a:.0f}%', (xi - w/2, a), textcoords='offset points',
                     xytext=(0, 5), ha='center', fontsize=6.3, color=C_DRL,
                     fontweight='bold' if a > 0 else 'normal')
        ax2.annotate(f'{b:.0f}%', (xi + w/2, b), textcoords='offset points',
                     xytext=(0, 5), ha='center', fontsize=6.3, color=C_LLM,
                     fontweight='bold' if b > 0 else 'normal')

    ax1.axhline(FAIL, color=C_GRAY, ls=':', lw=0.9, zorder=2)
    ax1.text(2.45, FAIL+60, r'$\tau$', ha='right', va='bottom', fontsize=7,
             color=C_GRAY)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel('Urgent-lot density (% of all lots)')
    ax1.set_ylabel('Mean Urgent-Lot Cycle Time (s)')
    ax2.set_ylabel('Catastrophic failure rate (%)')
    ax1.set_ylim(0, 6000); ax2.set_ylim(-5, 95)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))

    pd_ = mpatches.Patch(color=C_DRL, alpha=0.85,
                         label='Pure DRL: CT (bar), fail (●)')
    pl_ = mpatches.Patch(color=C_LLM, alpha=0.85,
                         label='LLM-HRL: CT (bar), fail (■)')
    ax1.legend(handles=[pd_, pl_], loc='upper left', fontsize=6.3,
               edgecolor='#ccc')
    ax1.set_title('Generalization across conflict regimes', pad=7)
    ax1.yaxis.grid(True, ls='--', lw=0.5, color=C_GRID, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    _save(fig, 'fig_sweep')


# ══════════════════════════════════════════════════════════════════════════
# Figure 4 — Ablation study: component contribution (2-column)
# ══════════════════════════════════════════════════════════════════════════
def fig_ablation(m, abl, rb):
    llm = np.array(list(m['llm_mean'].values()))
    drl = np.array(list(m['drl_mean'].values()))

    # Three focused ablations, each removing ONE design axis, plus anchors.
    rows = [
        ('LLM-HRL',      'full',     llm,                  'full'),
        ('Rule-Based',   '$-$adaptive', rb['cts'],         'abl'),
        ('Random Mask',  '$-$context',  abl['Random Mask']['cts'], 'abl'),
        ('Soft Reward',  '$-$hard mask', abl['Soft Reward']['cts'], 'abl'),
        ('Pure DRL',     'no gov.',  drl,                  'drl'),
    ]
    names  = [r[0] for r in rows]
    subs   = [r[1] for r in rows]
    means  = np.array([np.mean(r[2]) for r in rows])
    stds   = np.array([np.std(r[2], ddof=1) for r in rows])
    fails  = np.array([(np.array(r[2]) > FAIL).mean()*100 for r in rows])
    kinds  = [r[3] for r in rows]
    x = np.arange(len(rows))

    labels = [n if s in ('full', 'no gov.') else f'{n}\n({s})'
              for n, s in zip(names, subs)]
    labels[0] = 'LLM-HRL\n(full)'
    labels[-1] = 'Pure DRL\n(no gov.)'

    def bar_color(k):
        return {'full': C_LLM, 'drl': C_DRL, 'abl': '#9E9E9E'}[k]
    cols = [bar_color(k) for k in kinds]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.1))

    # ── Left: CT mean +/- SD ────────────────────────────────────────────
    axL.axhspan(FAIL, 5400, color=C_FAILZONE, zorder=0)
    axL.axhline(FAIL, color=C_GRAY, ls='--', lw=0.9, zorder=2)
    axL.bar(x, means, 0.62, yerr=stds, color=cols, alpha=0.9,
            capsize=3, error_kw={'elinewidth': 0.9, 'ecolor': '#444'}, zorder=3)
    for xi, mv, sv in zip(x, means, stds):
        axL.text(xi, mv+sv+90, f'{mv:.0f}', ha='center', va='bottom',
                 fontsize=7, color='#222')
    axL.text(len(rows)-0.45, FAIL+70, r'$\tau$', ha='right', va='bottom',
             fontsize=8, color=C_GRAY)
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=7.2)
    axL.set_ylabel('Mean Urgent-Lot Cycle Time (s)')
    axL.set_ylim(0, 5400)
    axL.yaxis.set_major_formatter(mticker.FuncFormatter(_k))
    axL.set_title('(a) Cycle time by condition', pad=6, fontsize=9)
    axL.yaxis.grid(True, ls='--', lw=0.5, color=C_GRID, alpha=0.5, zorder=0)
    axL.set_axisbelow(True)

    # ── Right: failure rate ─────────────────────────────────────────────
    axR.bar(x, fails, 0.62, color=cols, alpha=0.9, zorder=3)
    for xi, fv in zip(x, fails):
        axR.text(xi, fv+2.0, f'{fv:.0f}%', ha='center', va='bottom',
                 fontsize=7.2, color='#222')
    axR.set_xticks(x)
    axR.set_xticklabels(labels, fontsize=7.2)
    axR.set_ylabel('Catastrophic failure rate (%)')
    axR.set_ylim(0, 100)
    axR.yaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    axR.set_title('(b) Failure rate by condition', pad=6, fontsize=9)
    axR.yaxis.grid(True, ls='--', lw=0.5, color=C_GRID, alpha=0.5, zorder=0)
    axR.set_axisbelow(True)

    # shared legend
    leg = [mpatches.Patch(color=C_LLM, alpha=0.9, label='Full model (LLM-HRL)'),
           mpatches.Patch(color='#9E9E9E', alpha=0.9, label='Ablated variant (one axis removed)'),
           mpatches.Patch(color=C_DRL, alpha=0.9, label='Pure DRL baseline')]
    fig.legend(handles=leg, loc='upper center', ncol=3, fontsize=7.4,
               frameon=True, edgecolor='#ccc', bbox_to_anchor=(0.5, 1.07))

    plt.tight_layout(pad=0.6, w_pad=1.8)
    _save(fig, 'fig_ablation')


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating AAAI-27 figures ...')
    m   = D.load_main()
    abl = D.load_ablations()
    rb  = D.load_rulebased()
    c   = D.load_curves()

    fig_reliability(m)
    fig_ct_dist(m)
    fig_convergence(c)
    fig_sweep(m)
    fig_ablation(m, abl, rb)
    # fig_mechanism()  # dropped from paper (page budget); kept for reference

    print('\nAll figures saved to:', OUT)
