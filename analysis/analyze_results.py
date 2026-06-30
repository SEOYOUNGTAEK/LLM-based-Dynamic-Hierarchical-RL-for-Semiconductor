import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

"""
analyze_results.py
──────────────────
4개 실험이 모두 완료된 후 실행하여 논문용 비교 차트와 통계 테이블을 생성합니다.

사용법:
  python analyze_results.py

output/ 폴더의 results_*.csv 파일들을 자동으로 탐색합니다.
조건 파일 없이 CSV 경로를 직접 지정할 수도 있습니다.
"""

import os
import glob
import sys
import pandas as pd

# ── 분석 출력 디렉토리 ──────────────────────────────────────────
OUT_DIR = "output/paper_figures"

# ── CSV 자동 탐색 ────────────────────────────────────────────────
# 각 conditions_* CSV 파일에서 condition 키를 추출합니다.
CONDITION_CSV_PATTERNS = {
    # 핵심 4개 조건
    'PureDRL':          'output/**/results_PureDRL.csv',
    'OriginalLLM':      'output/**/results_Baseline_C_OriginalLLM.csv',
    'Phase1LLM':        'output/**/results_Baseline_B_Phase1LLM.csv',
    'Proposed_FullLLM': 'output/**/results_Proposed_FullLLM.csv',
    # 추가 비교 (있을 때만 포함)
    'RuleBased_LLM':    'output/**/results_RuleBased_LLM.csv',
    'Rule_FIFO':        'output/**/results_Rule_FIFO.csv',
    'Rule_EDD':         'output/**/results_Rule_EDD.csv',
}


def find_csv(pattern: str):
    """glob으로 가장 최근 파일 반환."""
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    from analysis.result_reporter.paper_visualizer import (
        compare_kpi_boxplot,
        compare_learning_curves,
        compare_ablation_bar,
        print_stats_table,
        analyze_convergence,
        draw_xai_rule_heatmap,
        draw_xai_reasoning_summary,
        draw_xai_action_distribution,
        draw_xai_engineer_intervention,
    )

    print("="*60)
    print("  LLM-HRL Experiment Analysis")
    print("="*60)

    # ── CSV 탐색 ─────────────────────────────────────────────────
    csv_paths = {}
    for key, pattern in CONDITION_CSV_PATTERNS.items():
        path = find_csv(pattern)
        if path:
            csv_paths[key] = path
            print(f"  Found [{key}]: {path}")
        else:
            print(f"  MISSING [{key}]: no file matching {pattern}")

    if len(csv_paths) < 2:
        print("\nNeed at least 2 conditions to compare. Exiting.")
        sys.exit(1)

    print(f"\n  Output directory: {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    print()

    # ── 1. KPI 박스플롯 ──────────────────────────────────────────
    print("[1/8] Generating KPI boxplot...")
    compare_kpi_boxplot(csv_paths, OUT_DIR)

    # ── 2. 학습 곡선 비교 ────────────────────────────────────────
    print("[2/8] Generating learning curves...")
    compare_learning_curves(csv_paths, OUT_DIR, eval_start=70)

    # ── 3. Ablation bar chart ────────────────────────────────────
    print("[3/8] Generating ablation bar chart...")
    compare_ablation_bar(csv_paths, OUT_DIR)

    # ── 4. 수렴 분석 ─────────────────────────────────────────────
    print("[4/8] Analyzing convergence speed...")
    conv_df = analyze_convergence(csv_paths, OUT_DIR, eval_start=70)

    # ── 5. 통계 테이블 (t-test) ──────────────────────────────────
    print("[5/8] Computing statistics table...")
    result_df = print_stats_table(csv_paths, OUT_DIR)

    # ── 6. XAI: 에피소드 수준 (LLM 추론 + 룰 히트맵) ─────────────
    print("[6/8] Generating XAI episode-level charts...")
    XAI_EPISODE_PATTERNS = {
        'OriginalLLM':      'output/**/*Baseline_C*/**/xai_decisions.json',
        'Phase1LLM':        'output/**/*Baseline_B*/**/xai_decisions.json',
        'Proposed_FullLLM': 'output/**/*Proposed*/**/xai_decisions.json',
        'RuleBased_LLM':    'output/**/*RuleBased*/**/xai_decisions.json',
    }
    for key, pattern in XAI_EPISODE_PATTERNS.items():
        xai_path = find_csv(pattern)
        if xai_path:
            print(f"  XAI-episode [{key}]: {xai_path}")
            draw_xai_rule_heatmap(xai_path, OUT_DIR + f'/{key}')
            draw_xai_reasoning_summary(xai_path, OUT_DIR + f'/{key}')
        else:
            print(f"  XAI-episode [{key}]: not found (skip)")

    # ── 7. XAI: 스텝 수준 (엔지니어 개입 추적) ───────────────────
    print("[7/8] Generating XAI step-level intervention charts...")
    XAI_STEP_PATTERNS = {
        'Proposed_FullLLM': 'output/**/*Proposed*/**/xai_step_interventions.json',
        'RuleBased_LLM':    'output/**/*RuleBased*/**/xai_step_interventions.json',
    }
    for key, pattern in XAI_STEP_PATTERNS.items():
        step_path = find_csv(pattern)
        if step_path:
            print(f"  XAI-step [{key}]: {step_path}")
            draw_xai_engineer_intervention(step_path, OUT_DIR + f'/{key}')
        else:
            print(f"  XAI-step [{key}]: not found (skip)")

    # ── 8. XAI: 거버넌스 강도 비교 ──────────────────────────────
    print("[8/8] Generating XAI governance intensity comparison...")
    draw_xai_action_distribution(csv_paths, OUT_DIR)

    # ── 최종 요약 ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  QUICK SUMMARY (Eval Episodes)")
    print("="*60)
    for key, path in csv_paths.items():
        df = pd.read_csv(path)
        eval_df = df[df['is_eval'] == True] if 'is_eval' in df.columns else df.tail(30)
        ct_mean = eval_df['AVG_ULOT_CYCLE_TIME'].mean()
        ct_std  = eval_df['AVG_ULOT_CYCLE_TIME'].std(ddof=1)
        tm_mean = eval_df['target_meet_rate'].mean()
        tm_std  = eval_df['target_meet_rate'].std(ddof=1)
        n = len(eval_df)
        label = key.ljust(22)
        print(f"  {label}  CT={ct_mean:>7,.1f}±{ct_std:>6,.1f}  "
              f"TM={tm_mean:>5.1f}%±{tm_std:.1f}  (n={n})")

    print(f"\n  All figures saved to: {OUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
