"""
Paper-quality visualization for LLM-HRL experiment results.

Per-experiment (called during training):
  - draw_kpi_trend()    : Cycle time + Target meet rate over episodes
  - draw_loss_curve()   : DQN training loss

Cross-experiment (called by analyze_results.py after all runs):
  - compare_kpi_boxplot()    : Eval episode KPI boxplots
  - compare_learning_curves(): Smoothed learning curves
  - compare_ablation_bar()   : Ablation bar chart
  - print_stats_table()      : t-test + mean/std table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── 공통 스타일 ────────────────────────────────────────────────
CONDITION_COLORS = {
    # ── Ablation 4종 ──
    'PureDRL':           '#4C72B0',   # blue
    'OriginalLLM':       '#DD8452',   # orange
    'Phase1LLM':         '#55A868',   # green
    'Proposed_FullLLM':  '#C44E52',   # red
    # ── 외부 비교군 3종 ──
    'RuleBased_LLM':     '#9467BD',   # purple  (LLM 없는 Rule-based ablation)
    'Rule_FIFO':         '#8C8C8C',   # gray    (단순 규칙 FIFO)
    'Rule_EDD':          '#BCBD22',   # yellow-green (단순 규칙 EDD)
}
CONDITION_LABELS = {
    # ── Ablation 4종 ──
    'PureDRL':           '(A) Pure DRL',
    'OriginalLLM':       '(B) Original LLM',
    'Phase1LLM':         '(C) Phase1 LLM\n(+Constraint)',
    'Proposed_FullLLM':  '(D) Proposed\n(Full LLM)',
    # ── 외부 비교군 3종 ──
    'RuleBased_LLM':     '(E) Rule-Based\nMock',
    'Rule_FIFO':         'FIFO',
    'Rule_EDD':          'EDD',
}
# 논문 ablation 순서 (A→B→C→D)
ABLATION_ORDER = ['PureDRL', 'OriginalLLM', 'Phase1LLM', 'Proposed_FullLLM']
# 전체 비교 순서 (통계 테이블 / boxplot)
FULL_ORDER = ['PureDRL', 'OriginalLLM', 'Phase1LLM', 'Proposed_FullLLM',
              'RuleBased_LLM', 'Rule_FIFO', 'Rule_EDD']
EVAL_BOUNDARY_COLOR = '#999999'
CONSTRAINT_LINE_COLOR = '#E74C3C'
ROLLING_WINDOW = 5
DPI = 150
FONTSIZE = 11

plt.rcParams.update({
    'font.size': FONTSIZE,
    'axes.titlesize': FONTSIZE + 1,
    'axes.labelsize': FONTSIZE,
    'legend.fontsize': FONTSIZE - 1,
    'figure.dpi': DPI,
})


# ════════════════════════════════════════════════════════════════
# 1. Per-experiment charts
# ════════════════════════════════════════════════════════════════

def draw_kpi_trend(results: list, eval_start: int, plot_dir: str, episode: int,
                   condition: str = 'Unknown'):
    """
    에피소드별 KPI 추세 (2-panel).
    Top: Urgent Lot Cycle Time + rolling mean
    Bottom: Target Meet Rate + 90% threshold line
    """
    if not results:
        return

    df = pd.DataFrame(results)
    ep = df['episode'].values
    ct = df['AVG_ULOT_CYCLE_TIME'].values.astype(float)
    tm = df['target_meet_rate'].values.astype(float)

    ct_roll = _rolling_mean(ct, ROLLING_WINDOW)
    tm_roll = _rolling_mean(tm, ROLLING_WINDOW)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f'KPI Trend — {_label(condition)}', fontweight='bold')

    # ─ Panel 1: Cycle Time ─
    ax1.plot(ep, ct, alpha=0.3, color='steelblue', linewidth=1, label='Per episode')
    ax1.plot(ep, ct_roll, color='steelblue', linewidth=2,
             label=f'Rolling mean (N={ROLLING_WINDOW})')
    ax1.axvline(eval_start + 0.5, color=EVAL_BOUNDARY_COLOR, linestyle='--',
                linewidth=1.5, label='Train / Eval boundary')
    ax1.set_ylabel('ULot Cycle Time (sec)')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    ax1.fill_betweenx([ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 0, max(ct) * 1.05],
                       eval_start + 0.5, max(ep) + 0.5,
                       alpha=0.05, color='green', label='_')

    # ─ Panel 2: Target Meet Rate ─
    ax2.plot(ep, tm, alpha=0.3, color='darkorange', linewidth=1)
    ax2.plot(ep, tm_roll, color='darkorange', linewidth=2,
             label=f'Rolling mean (N={ROLLING_WINDOW})')
    ax2.axhline(90, color=CONSTRAINT_LINE_COLOR, linestyle='--',
                linewidth=1.5, label='Constraint (90%)')
    ax2.axvline(eval_start + 0.5, color=EVAL_BOUNDARY_COLOR, linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Target Meet Rate (%)')
    ax2.set_ylim(70, 100)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(plot_dir, f'kpi_trend_ep{episode + 1}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def draw_reward_convergence(rewards: list, losses: list, eval_start: int,
                            plot_dir: str, episode: int, condition: str = 'Unknown'):
    """총 보상 수렴 + DQN loss 곡선 (2-panel)."""
    if not rewards:
        return

    ep = list(range(1, len(rewards) + 1))
    r_roll = _rolling_mean(rewards, ROLLING_WINDOW)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    fig.suptitle(f'Training Convergence — {_label(condition)}', fontweight='bold')

    # ─ Panel 1: Reward ─
    ax1.plot(ep, rewards, alpha=0.25, color='royalblue', linewidth=1)
    ax1.plot(ep, r_roll, color='royalblue', linewidth=2,
             label=f'Rolling mean (N={ROLLING_WINDOW})')
    ax1.axvline(eval_start + 0.5, color=EVAL_BOUNDARY_COLOR, linestyle='--',
                linewidth=1.5, label='Train / Eval')
    ax1.set_ylabel('Total Reward')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    # ─ Panel 2: Loss ─
    if losses:
        ax2.plot(losses, color='crimson', linewidth=0.8, alpha=0.6)
        loss_roll = _rolling_mean(losses, 50)
        ax2.plot(loss_roll, color='darkred', linewidth=1.5, label='Rolling mean (N=50)')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', linestyle=':', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, 'No loss data (eval-only)', ha='center', va='center',
                 transform=ax2.transAxes)

    plt.tight_layout()
    path = os.path.join(plot_dir, f'convergence_ep{episode + 1}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def draw_governance_activity(results: list, eval_start: int,
                             plot_dir: str, episode: int, condition: str = 'Unknown'):
    """LLM 거버넌스 활동 (마스킹 비율 + 룰 적용 횟수)."""
    df = pd.DataFrame(results)
    if 'masking_rate' not in df.columns or 'L2_rules_applied_cumul' not in df.columns:
        return

    ep = df['episode'].values
    mask_rate = df['masking_rate'].values * 100  # %

    fig, ax1 = plt.subplots(figsize=(10, 4))
    fig.suptitle(f'LLM Governance Activity — {_label(condition)}', fontweight='bold')

    color1 = 'steelblue'
    ax1.bar(ep, mask_rate, color=color1, alpha=0.5, label='Masking rate (%)')
    ax1.plot(ep, _rolling_mean(mask_rate, ROLLING_WINDOW),
             color=color1, linewidth=2, label=f'Rolling mean (N={ROLLING_WINDOW})')
    ax1.axvline(eval_start + 0.5, color=EVAL_BOUNDARY_COLOR, linestyle='--',
                linewidth=1.5, label='Train / Eval')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Step Masking Rate (%)', color=color1)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'darkorange'
    ax2.plot(ep, df['L2_rules_applied_cumul'].values,
             color=color2, linewidth=2, linestyle='-.', label='Cumul. L2 rules applied')
    ax2.set_ylabel('Cumulative L2 Calls', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    ax1.grid(axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    path = os.path.join(plot_dir, f'governance_ep{episode + 1}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# 2. Cross-experiment comparison (analyze_results.py 에서 호출)
# ════════════════════════════════════════════════════════════════

def _load_eval(csv_path: str) -> pd.DataFrame:
    """CSV 로드 후 eval 에피소드만 반환."""
    df = pd.read_csv(csv_path)
    if 'is_eval' in df.columns:
        return df[df['is_eval'] == True].copy()
    # fallback: 마지막 30 에피소드
    return df.tail(30).copy()


def _condition_key(csv_path: str) -> str:
    """CSV 파일 경로에서 조건 키 추출."""
    name = os.path.basename(csv_path)
    for key in CONDITION_COLORS:
        if key in name:
            return key
    return os.path.splitext(name)[0]


def analyze_convergence(csv_paths: dict, out_dir: str, eval_start: int = 70,
                        threshold_pct: float = 0.10):
    """
    수렴 분석: 각 조건이 몇 에피소드 만에 최종 성능의 (1+threshold)% 이내로 수렴했는지.
    → "LLM guidance는 DRL 수렴 속도를 X에피소드 단축시킨다"
    """
    os.makedirs(out_dir, exist_ok=True)
    order = ['PureDRL', 'OriginalLLM', 'Phase1LLM', 'Proposed_FullLLM',
             'RuleBased_LLM', 'Rule_FIFO', 'Rule_EDD']

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Convergence Analysis — Episodes to Reach Stable Performance',
                 fontweight='bold')

    conv_results = []
    for key in order:
        if key not in csv_paths:
            continue
        df = pd.read_csv(csv_paths[key])
        train_df = df[df['is_eval'] == False] if 'is_eval' in df.columns else df.head(eval_start)
        if train_df.empty:
            continue

        ct = train_df['AVG_ULOT_CYCLE_TIME'].values.astype(float)
        # 최종 10에피소드 평균 = 수렴 목표값
        final_mean = np.mean(ct[-10:]) if len(ct) >= 10 else np.mean(ct)
        target = final_mean * (1 + threshold_pct)

        # 처음으로 target 이하로 떨어진 에피소드
        conv_ep = None
        for i in range(len(ct)):
            if ct[i] <= target:
                conv_ep = i + 1
                break

        color = CONDITION_COLORS.get(key, 'gray')
        label = CONDITION_LABELS.get(key, key).replace('\n', ' ')
        ep = list(range(1, len(ct) + 1))
        ax.plot(ep, _rolling_mean(ct, ROLLING_WINDOW), color=color,
                linewidth=2, label=label, alpha=0.8)

        conv_results.append({
            'condition': key,
            'label': label,
            'convergence_episode': conv_ep,
            'final_ct_mean': round(final_mean, 1),
        })

    ax.axvline(eval_start + 0.5, color=EVAL_BOUNDARY_COLOR, linestyle='--',
               linewidth=1.5, label='Train / Eval boundary')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('ULot Cycle Time (sec, rolling mean)')
    ax.set_title(f'Convergence (threshold: within {threshold_pct*100:.0f}% of final value)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, 'convergence_analysis.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

    # 수렴 결과 테이블 저장
    conv_df = pd.DataFrame(conv_results)
    conv_df.to_csv(os.path.join(out_dir, 'convergence_table.csv'), index=False)

    print(f'  Saved: {path}')
    if not conv_df.empty:
        print('\n  Convergence Summary:')
        for _, row in conv_df.iterrows():
            ep_str = f"Ep {row['convergence_episode']}" if row['convergence_episode'] else "Not converged"
            print(f"    {row['label'][:30]:<30} → {ep_str}  (final CT={row['final_ct_mean']:,.0f})")
    return conv_df


def compare_kpi_boxplot(csv_paths: dict, out_dir: str):
    """
    Eval 에피소드 KPI 박스플롯 (논문 Figure 4 대체).
    csv_paths: {condition_key: csv_path}
    """
    os.makedirs(out_dir, exist_ok=True)

    # FULL_ORDER 기준으로 정렬 (없는 조건은 뒤에)
    ordered_keys = [k for k in FULL_ORDER if k in csv_paths]
    ordered_keys += [k for k in csv_paths if k not in FULL_ORDER]

    conditions, ct_data, tm_data, colors, labels = [], [], [], [], []
    n_eval_list = []
    for key in ordered_keys:
        path = csv_paths[key]
        df = _load_eval(path)
        conditions.append(key)
        ct_data.append(df['AVG_ULOT_CYCLE_TIME'].dropna().values)
        tm_data.append(df['target_meet_rate'].dropna().values)
        colors.append(CONDITION_COLORS.get(key, 'gray'))
        labels.append(CONDITION_LABELS.get(key, key).replace('\n', '\n'))
        n_eval_list.append(len(df))

    n_eval_str = n_eval_list[0] if len(set(n_eval_list)) == 1 else f'{min(n_eval_list)}~{max(n_eval_list)}'
    n_ablation = sum(1 for k in conditions if k in ABLATION_ORDER)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(conditions) * 1.6), 6))
    fig.suptitle(f'KPI Comparison — Evaluation Episodes (n={n_eval_str} per condition)',
                 fontweight='bold')

    # ─ Cycle Time boxplot ─
    bp1 = ax1.boxplot(ct_data, patch_artist=True, widths=0.5,
                      medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xticks(range(1, len(labels) + 1))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('ULot Cycle Time (sec)')
    ax1.set_title('Urgent Lot Cycle Time')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    # 평균값 표시
    for i, data in enumerate(ct_data):
        ax1.scatter(i + 1, np.mean(data), marker='D', color='black', s=40, zorder=5)

    # ─ Target Meet Rate boxplot ─
    bp2 = ax2.boxplot(tm_data, patch_artist=True, widths=0.5,
                      medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(90, color=CONSTRAINT_LINE_COLOR, linestyle='--',
                linewidth=1.5, label='Constraint (90%)')
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Target Meet Rate (%)')
    ax2.set_title('Target Meet Rate')
    ax2.set_ylim(75, 100)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    for i, data in enumerate(tm_data):
        ax2.scatter(i + 1, np.mean(data), marker='D', color='black', s=40, zorder=5)

    plt.tight_layout()
    path = os.path.join(out_dir, 'compare_kpi_boxplot.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def compare_learning_curves(csv_paths: dict, out_dir: str, eval_start: int = 70):
    """
    4개 조건 학습 곡선 비교 (cycle time rolling mean).
    """
    os.makedirs(out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Learning Curves — All Conditions', fontweight='bold')

    for key, path in csv_paths.items():
        df = pd.read_csv(path)
        ep = df['episode'].values
        ct = df['AVG_ULOT_CYCLE_TIME'].values.astype(float)
        tm = df['target_meet_rate'].values.astype(float)
        ct_roll = _rolling_mean(ct, ROLLING_WINDOW)
        tm_roll = _rolling_mean(tm, ROLLING_WINDOW)
        color = CONDITION_COLORS.get(key, 'gray')
        label = CONDITION_LABELS.get(key, key).replace('\n', ' ')

        ax1.plot(ep, ct, alpha=0.12, color=color, linewidth=1)
        ax1.plot(ep, ct_roll, color=color, linewidth=2, label=label)

        ax2.plot(ep, tm, alpha=0.12, color=color, linewidth=1)
        ax2.plot(ep, tm_roll, color=color, linewidth=2, label=label)

    for ax in (ax1, ax2):
        ax.axvline(eval_start + 0.5, color=EVAL_BOUNDARY_COLOR, linestyle='--',
                   linewidth=1.5, label='Train / Eval boundary')

    ax1.set_ylabel('ULot Cycle Time (sec)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    ax2.axhline(90, color=CONSTRAINT_LINE_COLOR, linestyle='--',
                linewidth=1.5, label='Constraint (90%)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Target Meet Rate (%)')
    ax2.set_ylim(70, 100)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, 'compare_learning_curves.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def compare_ablation_bar(csv_paths: dict, out_dir: str):
    """
    Ablation bar chart: (A)→(D) ablation + 비교군(FIFO/EDD/RuleBased) 점선으로 표시.
    """
    os.makedirs(out_dir, exist_ok=True)
    order = ABLATION_ORDER  # ablation 4종만 bar로

    ct_means, ct_stds, tm_means, tm_stds = [], [], [], []
    valid_order, valid_labels, valid_colors = [], [], []

    for key in order:
        if key not in csv_paths:
            continue
        df = _load_eval(csv_paths[key])
        ct_vals = df['AVG_ULOT_CYCLE_TIME'].dropna().values
        tm_vals = df['target_meet_rate'].dropna().values
        ct_means.append(np.mean(ct_vals))
        ct_stds.append(np.std(ct_vals, ddof=1))
        tm_means.append(np.mean(tm_vals))
        tm_stds.append(np.std(tm_vals, ddof=1))
        valid_order.append(key)
        valid_labels.append(CONDITION_LABELS.get(key, key).replace('\n', '\n'))
        valid_colors.append(CONDITION_COLORS.get(key, 'gray'))

    x = np.arange(len(valid_order))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ablation Study — Progressive LLM Governance Improvements',
                 fontweight='bold')

    # ─ Cycle Time ─
    bars1 = ax1.bar(x, ct_means, yerr=ct_stds, capsize=6,
                    color=valid_colors, alpha=0.75, edgecolor='black', linewidth=0.8,
                    error_kw=dict(elinewidth=1.5, ecolor='black'))
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_labels, fontsize=9)
    ax1.set_ylabel('ULot Cycle Time (sec)')
    ax1.set_title('Urgent Lot Cycle Time\n(lower is better)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    # % 개선 표시
    baseline = ct_means[0]
    for i, (mean, std) in enumerate(zip(ct_means, ct_stds)):
        improv = (baseline - mean) / baseline * 100
        label = f'{mean:,.0f}\n(±{std:.0f})'
        if i > 0:
            label += f'\n↓{improv:.1f}%'
        ax1.text(i, mean + std * 1.3, label, ha='center', va='bottom', fontsize=8)

    # ─ Target Meet Rate ─
    bars2 = ax2.bar(x, tm_means, yerr=tm_stds, capsize=6,
                    color=valid_colors, alpha=0.75, edgecolor='black', linewidth=0.8,
                    error_kw=dict(elinewidth=1.5, ecolor='black'))
    ax2.axhline(90, color=CONSTRAINT_LINE_COLOR, linestyle='--',
                linewidth=1.5, label='Constraint (90%)')

    # 비교군(FIFO/EDD/RuleBased) 수평 점선으로 참조선 추가
    ref_keys = ['RuleBased_LLM', 'Rule_FIFO', 'Rule_EDD']
    ref_linestyles = [':', '-.', '--']
    for rk, ls in zip(ref_keys, ref_linestyles):
        if rk in csv_paths:
            ref_df = _load_eval(csv_paths[rk])
            ref_tm = ref_df['target_meet_rate'].dropna().mean()
            ref_ct = ref_df['AVG_ULOT_CYCLE_TIME'].dropna().mean()
            ref_label = CONDITION_LABELS.get(rk, rk).replace('\n', ' ')
            ax2.axhline(ref_tm, color=CONDITION_COLORS.get(rk, 'gray'),
                        linestyle=ls, linewidth=1.2, alpha=0.8, label=f'{ref_label} ({ref_tm:.1f}%)')
            ax1.axhline(ref_ct, color=CONDITION_COLORS.get(rk, 'gray'),
                        linestyle=ls, linewidth=1.2, alpha=0.8, label=f'{ref_label} ({ref_ct:,.0f})')

    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_labels, fontsize=9)
    ax2.set_ylabel('Target Meet Rate (%)')
    ax2.set_title('Target Meet Rate\n(↑ higher is better)')
    ax2.set_ylim(75, 100)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    for i, (mean, std) in enumerate(zip(tm_means, tm_stds)):
        ax2.text(i, mean + std * 1.3, f'{mean:.1f}%\n(±{std:.1f})',
                 ha='center', va='bottom', fontsize=8)

    ax1.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    path = os.path.join(out_dir, 'compare_ablation_bar.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def print_stats_table(csv_paths: dict, out_dir: str):
    """
    통계 분석 테이블 (mean ± std, t-test vs Pure DRL).
    CSV로 저장 + 콘솔 출력.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Ablation 4종 + 비교군 3종 전체 포함
    order = FULL_ORDER
    baseline_ct, baseline_tm = None, None
    rows = []

    for key in order:
        if key not in csv_paths:
            continue
        df = _load_eval(csv_paths[key])
        ct = df['AVG_ULOT_CYCLE_TIME'].dropna().values
        tm = df['target_meet_rate'].dropna().values

        ct_mean, ct_std = np.mean(ct), np.std(ct, ddof=1)
        tm_mean, tm_std = np.mean(tm), np.std(tm, ddof=1)

        if key == 'PureDRL':
            baseline_ct, baseline_tm = ct, tm
            ct_improv, tm_improv = '-', '-'
            ct_pval, tm_pval = '-', '-'
        else:
            ct_improv = f'{(np.mean(baseline_ct) - ct_mean) / np.mean(baseline_ct) * 100:.2f}%'
            tm_improv = f'{tm_mean - np.mean(baseline_tm):+.2f}pp'
            _, p_ct = stats.ttest_ind(baseline_ct, ct, equal_var=False)
            _, p_tm = stats.ttest_ind(baseline_tm, tm, equal_var=False)
            ct_pval = f'{p_ct:.4f}' + (' *' if p_ct < 0.05 else '')
            tm_pval = f'{p_tm:.4f}' + (' *' if p_tm < 0.05 else '')

        rows.append({
            'Condition': CONDITION_LABELS.get(key, key).replace('\n', ' '),
            'CycleTime_Mean': f'{ct_mean:,.1f}',
            'CycleTime_Std':  f'{ct_std:,.1f}',
            'CycleTime_Impr': ct_improv,
            'CycleTime_pval': ct_pval,
            'TargetMeet_Mean': f'{tm_mean:.2f}%',
            'TargetMeet_Std':  f'{tm_std:.2f}',
            'TargetMeet_Impr': tm_improv,
            'TargetMeet_pval': tm_pval,
            'n_eval': len(ct),
        })

    result_df = pd.DataFrame(rows)
    csv_out = os.path.join(out_dir, 'stats_table.csv')
    result_df.to_csv(csv_out, index=False)

    print('\n' + '='*80)
    print('PAPER STATISTICS TABLE  (Welch t-test vs Pure DRL, * p<0.05)')
    print('='*80)
    print(result_df.to_string(index=False))
    print('='*80)
    print(f'  Saved: {csv_out}')
    return result_df


# ════════════════════════════════════════════════════════════════
# 3. XAI Visualizations
# ════════════════════════════════════════════════════════════════

def draw_xai_rule_heatmap(xai_json_path: str, out_dir: str):
    """
    에피소드 × 액션 마스킹 히트맵.
    어떤 에피소드에서 어떤 액션이 금지됐는지 한눈에 보여줌.
    → 논문 XAI 섹션 핵심 그림
    """
    import json as _json
    os.makedirs(out_dir, exist_ok=True)

    with open(xai_json_path, encoding='utf-8') as f:
        records = _json.load(f)

    if not records:
        return

    episodes = [r.get('episode', i+1) for i, r in enumerate(records)]
    n_ep = max(episodes)
    n_actions = 10

    # 히트맵 행렬: episodes × actions
    heatmap = np.zeros((n_ep, n_actions))
    for r in records:
        ep = r.get('episode', 0) - 1
        if ep < 0 or ep >= n_ep:
            continue
        for a in r.get('masked_actions', []):
            if 0 <= a < n_actions:
                heatmap[ep, a] = 1

    fig, ax = plt.subplots(figsize=(12, max(4, n_ep * 0.18)))
    im = ax.imshow(heatmap.T, aspect='auto', cmap='YlOrRd',
                   origin='lower', vmin=0, vmax=1)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Action Index')
    ax.set_title('LLM Action Masking Timeline\n(Red = Forbidden by L2 Rule)')
    ax.set_yticks(range(n_actions))
    ax.set_yticklabels([
        'A0: IPLT-time(1)', 'A1: IPLT-time(2)', 'A2: IPLT-time(3)',
        'A3: Target(1)', 'A4: Target(2)', 'A5: Target(3)',
        'A6: IPLT-mgmt(1)', 'A7: IPLT-mgmt(2)', 'A8: IPLT-mgmt(3)',
        'A9: ULot Priority'
    ], fontsize=8)
    ax.set_xticks(range(0, n_ep, max(1, n_ep // 10)))

    plt.colorbar(im, ax=ax, label='Masked (1=Yes)', shrink=0.6)
    plt.tight_layout()
    path = os.path.join(out_dir, 'xai_rule_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def draw_xai_reasoning_summary(xai_json_path: str, out_dir: str, top_n: int = 3):
    """
    대표 추론 케이스 추출 + 텍스트 시각화.
    논문 Section 5.3 Qualitative Example에 바로 사용 가능.
    """
    import json as _json
    os.makedirs(out_dir, exist_ok=True)

    with open(xai_json_path, encoding='utf-8') as f:
        records = _json.load(f)

    # 룰이 생성된 에피소드 중 cycle time 변화가 가장 큰 케이스 선택
    rule_eps = [r for r in records if r.get('n_rules', 0) > 0 and r.get('reasoning')]
    if not rule_eps:
        return

    # cycle time 기준 정렬 (낮을수록 개선됨 → 흥미로운 케이스)
    rule_eps_sorted = sorted(rule_eps,
                             key=lambda x: x.get('AVG_ULOT_CYCLE_TIME', 9999))
    top_cases = rule_eps_sorted[:top_n]

    fig, axes = plt.subplots(top_n, 1, figsize=(14, 4 * top_n))
    if top_n == 1:
        axes = [axes]
    fig.suptitle('LLM Strategic Reasoning Examples (XAI)\nRepresentative Cases Selected by Cycle Time',
                 fontweight='bold', fontsize=12)

    for ax, case in zip(axes, top_cases):
        ep = case.get('episode', '?')
        ct = case.get('AVG_ULOT_CYCLE_TIME', 0)
        tm = case.get('target_meet_rate', 0)
        n_rules = case.get('n_rules', 0)
        masked = case.get('masked_actions', [])
        reasoning = case.get('reasoning', 'N/A')
        critic = case.get('critic_analysis', 'N/A')

        # 텍스트 패널
        context = (f"Episode {ep}  |  Cycle Time: {ct:,.0f}s  |  "
                   f"Target Meet: {tm:.1f}%  |  Rules: {n_rules}  |  Masked: {masked}")
        body = (f"[REASONING]\n{reasoning[:300]}{'...' if len(reasoning)>300 else ''}\n\n"
                f"[CRITIC]\n{critic[:200]}{'...' if len(critic)>200 else ''}")

        ax.axis('off')
        ax.text(0.01, 0.95, context, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='#E8F4FD', alpha=0.8))
        ax.text(0.01, 0.75, body, transform=ax.transAxes,
                fontsize=8, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#FAFAFA', alpha=0.8),
                wrap=True)

    plt.tight_layout()
    path = os.path.join(out_dir, 'xai_reasoning_examples.png')
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'  Saved: {path}')


def draw_xai_engineer_intervention(xai_step_path: str, out_dir: str, max_cases: int = 5):
    """
    엔지니어 XAI: LLM이 DRL을 오버라이드한 개입 사례 분석.

    보여주는 것:
    1. DRL이 원래 선택하려 했던 액션 vs LLM이 강제한 액션
    2. 개입 시점의 상황 (긴급 Lot 수, 생산 부족률)
    3. 어떤 규칙이 왜 트리거됐는지

    → 논문 Fig: "Why did the system choose Action 9 instead of Action 3?"
    """
    import json as _json
    os.makedirs(out_dir, exist_ok=True)

    with open(xai_step_path, encoding='utf-8') as f:
        records = _json.load(f)

    # 실제로 DRL 판단을 바꾼 케이스만 필터
    override_cases = [r for r in records if r.get('llm_overrode', False)]
    if not override_cases:
        # 개입은 있었지만 DRL이 이미 같은 선택을 한 케이스
        override_cases = records[:max_cases] if records else []

    if not override_cases:
        return

    # eval 에피소드 케이스 우선 (최종 정책 평가)
    eval_cases = [r for r in override_cases if r.get('is_eval', False)]
    cases = (eval_cases if eval_cases else override_cases)[:max_cases]

    ACTION_NAMES = {
        0: 'A0: IPLT(1)', 1: 'A1: IPLT(2)', 2: 'A2: IPLT(3)',
        3: 'A3: Target(1)', 4: 'A4: Target(2)', 5: 'A5: Target(3)',
        6: 'A6: IPLT-Mgmt(1)', 7: 'A7: IPLT-Mgmt(2)', 8: 'A8: IPLT-Mgmt(3)',
        9: 'A9: ULot Priority'
    }

    n = len(cases)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.5 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle('Engineer XAI: LLM Intervention Decision Traces\n'
                 '"Why was Action X chosen instead of Action Y?"',
                 fontweight='bold', fontsize=12)

    for row, (ax_bar, ax_text), case in zip(range(n), axes, cases):
        ep = case.get('episode', '?')
        sim_t = case.get('simTime', 0)
        drl_pref = case.get('drl_preferred_action', -1)
        drl_qval = case.get('drl_preferred_qval', 0)
        final_act = case.get('final_action', -1)
        top3 = case.get('drl_top3', [])
        rules = case.get('triggered_rules', [])
        state_sum = case.get('state_summary', '')

        # ─ Left: Q값 막대 그래프 ─
        actions = [a for a, _ in top3]
        qvals   = [q for _, q in top3]
        colors  = ['#E74C3C' if a == drl_pref else '#95A5A6' for a in actions]
        bars = ax_bar.barh([ACTION_NAMES.get(a, f'A{a}') for a in actions],
                           qvals, color=colors, alpha=0.8, edgecolor='black')
        if final_act != drl_pref:
            ax_bar.axhline(
                [ACTION_NAMES.get(a, f'A{a}') for a in actions].index(
                    ACTION_NAMES.get(final_act, f'A{final_act}')) if final_act in actions else -1,
                color='#27AE60', linewidth=2, linestyle='--')
        ax_bar.set_xlabel('Q-Value')
        ax_bar.set_title(f'Ep {ep} | simTime={sim_t:,.0f}s\nDRL Top-3 Q-Values',
                         fontsize=9)
        ax_bar.grid(axis='x', linestyle=':', alpha=0.5)

        # DRL 선호 vs 최종 선택 표시
        legend_elements = [
            mpatches.Patch(color='#E74C3C', label=f'DRL preferred: {ACTION_NAMES.get(drl_pref,"?")} (Q={drl_qval:.2f})'),
            mpatches.Patch(color='#27AE60', label=f'Final (LLM): {ACTION_NAMES.get(final_act,"?")}'),
        ]
        ax_bar.legend(handles=legend_elements, fontsize=7, loc='lower right')

        # ─ Right: 개입 이유 텍스트 ─
        ax_text.axis('off')
        rule_text = ""
        for rule in rules:
            rule_text += (f"  Rule: {rule.get('reason', '?')}\n"
                          f"  → Forbidden: {rule.get('forbidden_actions', [])}\n\n")

        state_short = '\n  '.join(state_sum.split('\n')[:3])
        full_text = (
            f"[Situation]\n  {state_short}\n\n"
            f"[L2 Rule Triggered]\n{rule_text if rule_text else '  No rules active'}"
            f"[Decision]\n"
            f"  DRL wanted:  {ACTION_NAMES.get(drl_pref, '?')} (Q={drl_qval:.2f})\n"
            f"  LLM blocked: {ACTION_NAMES.get(drl_pref, '?')}\n"
            f"  Final action: {ACTION_NAMES.get(final_act, '?')}\n"
            f"  Override: {'YES ← LLM intervened' if final_act != drl_pref else 'No (DRL agreed)'}"
        )
        ax_text.text(0.02, 0.95, full_text, transform=ax_text.transAxes,
                     fontsize=8, va='top', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(out_dir, 'xai_engineer_intervention.png')
    plt.savefig(path, bbox_inches='tight', dpi=120)
    plt.close(fig)
    print(f'  Saved: {path}')


def draw_xai_action_distribution(csv_paths: dict, out_dir: str):
    """
    조건별 액션 선택 분포 비교.
    LLM이 특정 액션(A9)을 얼마나 유도했는지 보여줌.
    """
    # rewards_data.csv에는 action 분포가 없으므로, masking_rate를 대리 지표로 사용
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('LLM Governance Effect on Policy Behavior', fontweight='bold')

    for key, path in csv_paths.items():
        df = pd.read_csv(path)
        color = CONDITION_COLORS.get(key, 'gray')
        label = CONDITION_LABELS.get(key, key).replace('\n', ' ')
        ep = df['episode'].values
        mr = df['masking_rate'].fillna(0).values * 100

        axes[0].plot(ep, mr, alpha=0.4, color=color, linewidth=1)
        axes[0].plot(ep, _rolling_mean(mr, ROLLING_WINDOW),
                     color=color, linewidth=2, label=label)

        # Eval 에피소드 평균
        eval_df = df[df['is_eval'] == True] if 'is_eval' in df.columns else df.tail(30)
        eval_mr = eval_df['masking_rate'].fillna(0).mean() * 100
        pr = eval_df['priority_process_rate'].fillna(0).mean()

    conditions = list(csv_paths.keys())
    eval_mask_rates = []
    eval_priority_rates = []
    for key, path in csv_paths.items():
        df = pd.read_csv(path)
        eval_df = df[df['is_eval'] == True] if 'is_eval' in df.columns else df.tail(30)
        eval_mask_rates.append(eval_df['masking_rate'].fillna(0).mean() * 100)
        eval_priority_rates.append(eval_df['priority_process_rate'].fillna(0).mean())

    x = np.arange(len(conditions))
    colors = [CONDITION_COLORS.get(k, 'gray') for k in conditions]
    labels = [CONDITION_LABELS.get(k, k).replace('\n', '\n') for k in conditions]

    axes[0].axvline(70.5, color=EVAL_BOUNDARY_COLOR, linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Step Masking Rate (%)')
    axes[0].set_title('Action Masking Rate Over Training')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', linestyle=':', alpha=0.5)

    bars = axes[1].bar(x, eval_mask_rates, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel('Avg Masking Rate (%) — Eval Episodes')
    axes[1].set_title('LLM Intervention Intensity\n(Eval Episodes Only)')
    axes[1].grid(axis='y', linestyle=':', alpha=0.5)
    for i, v in enumerate(eval_mask_rates):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'xai_governance_intensity.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ── 내부 유틸 ─────────────────────────────────────────────────
def _rolling_mean(data, window):
    result = np.full(len(data), np.nan)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result[i] = np.mean(data[start:i + 1])
    return result


def _label(condition):
    return CONDITION_LABELS.get(condition, condition).replace('\n', ' ')
