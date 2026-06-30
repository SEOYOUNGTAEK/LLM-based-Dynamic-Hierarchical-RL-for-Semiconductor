"""
train_baseline_rule.py
──────────────────────
전통 스케줄링 규칙 베이스라인 (DRL/LLM 없음).
고정 액션 정책으로만 평가 → 논문 비교 기준 강화

액션 의미:
  FIFO (Action 0): 체류 시간 기준 (가장 오래 기다린 Lot 우선) ≈ FCFS
  EDD  (Action 6): 잔여 IPLT 시간 기준 (마감 임박 순) ≈ Earliest Due Date

실행: python train_baseline_rule.py
RULE_ACTION = 0  → FIFO
RULE_ACTION = 6  → EDD
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from logger import Logger
import numpy as np
import torch
import sys
import pandas as pd
import os
import json
import time
import pickle
from datetime import datetime
from utils import NumpyJSONEncoder
from factory_data_generator.data_generator import Generator
from result_reporter.chart_generator import ChartGenerator
from twin import PlantsimManager

#####################
#   Rule Baseline   #
#####################
RULE_ACTION   = 0        # 0 = FIFO proxy, 6 = EDD  ← 여기서 변경
RULE_NAME     = "FIFO"   # "FIFO" or "EDD"          ← 여기서 변경

# FIFO: Action 0 (체류 시간 DESC = 오래 기다린 Lot 우선 ≈ FCFS)
# EDD : Action 6 (잔여 IPLT ASC  = 마감 임박 순 ≈ EDD)

NUM_EVAL_EPISODES = 30   # 평가만, 훈련 없음
RANDOM_SEED = 42
chart_frequency = 10
performance_frequency = 1
simulation_end_time = 86400

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def print_progress_bar(simTime, episode, total_episode):
    bar_length = 50
    progress = simTime / (simulation_end_time - 2000)
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\repisode {episode + 1}/{total_episode}: |{bar}|")
    sys.stdout.flush()


def analyze_iplt_and_target(target_df, output_df):
    output_qty_map = output_df.set_index("device_id")["qty"].to_dict()
    target_df["qty"] = target_df["device_id"].map(output_qty_map).fillna(0)
    target_df["achievement_rate"] = np.where(
        target_df["mat_qty"] > 0, (target_df["qty"] / target_df["mat_qty"]) * 100, 0)
    return target_df


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.set_printoptions(suppress=True, precision=6)

    model_type = f"Rule_{RULE_NAME}"
    version_no = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"output/{version_no}_{model_type}"
    log_dir = "./" + folder_name
    plot_dir = log_dir + '/plot/'
    for d in [folder_name, folder_name+"/csv", folder_name+"/log", plot_dir]:
        os.makedirs(d, exist_ok=True)

    logger = Logger(log_dir + '/log/' + version_no + '.log').get_logger()
    logger.info(f'=== Rule Baseline: {RULE_NAME} (Action {RULE_ACTION}) ===')
    logger.info(f'Eval episodes: {NUM_EVAL_EPISODES}  Seed: {RANDOM_SEED}')
    logger.info(f'No DRL training. No LLM. Fixed dispatching rule only.')

    generator = Generator()
    generator.run()

    all_episode_results = []
    plantsim_manager = PlantsimManager()
    plantsim_manager.init(False, version='23.2', visible='False', license_type='Educational')
    plantsim_manager.initialize_simulation(
        r"E:\SYT\SID_1031\SID_LLM_1031\iplt\IPLT.spp", folder_name, NUM_EVAL_EPISODES)

    for episode in range(NUM_EVAL_EPISODES):
        episode_start = time.time()
        logger.info(f"Episode {episode+1}/{NUM_EVAL_EPISODES} [EVAL — {RULE_NAME}]")
        plantsim_manager.start_simulation(episode)

        total_reward = 0
        total_target = 0
        total_urgent_lot = 0
        total_lot_move = 0
        total_priority_lot = 0
        unique_available = set()
        unique_processed = set()
        simTime = 0
        is_first = True
        wip_log = []
        last_wip_t = 0

        while True:
            if plantsim_manager.get_value("is_done"):
                break
            if simTime > simulation_end_time:
                break

            plantsim_manager.get_state()
            while plantsim_manager.get_value("WaitingSchedule"):
                simTime = plantsim_manager.get_sim_time()
                if simTime > last_wip_t + 3600:
                    wip_log.append(len(plantsim_manager.lot_state))
                    last_wip_t = simTime
                if simTime > simulation_end_time:
                    break

                # 고정 정책 — RULE_ACTION만 항상 선택
                action = RULE_ACTION

                plantsim_manager.execute_action(action, simTime)

                while not plantsim_manager.get_value("WaitingSchedule"):
                    time.sleep(0.1)
                    if plantsim_manager.get_value("is_done"):
                        break
                    if simTime > simulation_end_time:
                        break

                simTime = plantsim_manager.get_sim_time()
                next_state = plantsim_manager.get_state()

                reward, target, lot_move, urgent_lot_move, priority_lot_reward, _, \
                    _, avail_ids, proc_ids = plantsim_manager.calculate_reward(False)

                total_reward += reward
                total_target += target
                total_urgent_lot += urgent_lot_move
                total_lot_move += lot_move
                total_priority_lot += priority_lot_reward
                unique_available.update(avail_ids)
                unique_processed.update(proc_ids)

            print_progress_bar(simTime, episode, NUM_EVAL_EPISODES)

        # 에피소드 집계
        lot_history = plantsim_manager.get_value("var_lot_history")
        lot_history_df = pd.DataFrame(json.loads(lot_history))
        output_df = plantsim_manager.get_output_state()
        target_df = plantsim_manager.get_target_state()
        target_df = analyze_iplt_and_target(target_df, output_df)
        repair_df = plantsim_manager.get_repair_qty()

        total_production = (target_df["qty"] / 20).sum()
        iplt_exceed = repair_df["count"].sum()
        target_achievement = (target_df["qty"].sum() / target_df["mat_qty"].sum()) * 100
        avg_wip = np.mean(wip_log) if wip_log else 0

        ulots_df = lot_history_df[lot_history_df['lot_id'].str.startswith('U')].copy()
        if not ulots_df.empty:
            ulot_timing = ulots_df.groupby('lot_id')['sim_time'].agg(['min', 'max']).reset_index()
            avg_ulot_ct = (ulot_timing['max'] - ulot_timing['min']).mean()
        else:
            avg_ulot_ct = 0.0

        total_moved = len(unique_processed)
        total_avail = len(unique_available)
        priority_rate = (total_moved / total_avail * 100) if total_avail > 0 else 0

        episode_summary = {
            'episode':   episode + 1,
            'condition': model_type,
            'is_eval':   True,
            'seed':      RANDOM_SEED,
            'AVG_ULOT_CYCLE_TIME': avg_ulot_ct,
            'target_meet_rate':    target_achievement,
            'iplt_over_count':     int(iplt_exceed),
            'total_reward':    total_reward,
            'priority_reward': total_priority_lot,
            'L2_rules_applied_cumul': 0,
            'masking_rate': 0.0,
            'total_outputs':       float(total_production),
            'priority_process_rate': priority_rate,
            'avg_wip':             avg_wip,
            'total_action_steps':  0,
        }
        all_episode_results.append(episode_summary)

        elapsed = time.time() - episode_start
        logger.info(f'ep {episode+1} elapsed={elapsed:.1f}s  CT={avg_ulot_ct:,.1f}  TM={target_achievement:.1f}%')

    # 차트 + CSV 저장
    if all_episode_results:
        chart_generator = ChartGenerator(plot_dir, condition=model_type, eval_start=0)
        chart_generator.draw_kpi_trend(all_episode_results, NUM_EVAL_EPISODES - 1)

    results_df = pd.DataFrame(all_episode_results)
    results_df.to_csv(folder_name + "/csv/rewards_data.csv", index=False)
    results_df.to_csv(folder_name + f"/csv/results_{model_type}.csv", index=False)

    # 요약 출력
    ct_mean = results_df['AVG_ULOT_CYCLE_TIME'].mean()
    ct_std  = results_df['AVG_ULOT_CYCLE_TIME'].std(ddof=1)
    tm_mean = results_df['target_meet_rate'].mean()
    tm_std  = results_df['target_meet_rate'].std(ddof=1)
    print(f'\n=== {RULE_NAME} Results ({NUM_EVAL_EPISODES} episodes) ===')
    print(f'  Cycle Time: {ct_mean:,.1f} ± {ct_std:,.1f} sec')
    print(f'  Target Meet: {tm_mean:.1f} ± {tm_std:.1f} %')

    plantsim_manager.quit()
