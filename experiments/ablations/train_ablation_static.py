import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))

# train_ablation_static.py
# Ablation 3: Dynamic vs Static Rules
#
# 제안 모델과 동일한 LLM 룰 생성 + Hard Mask를 유지하되,
# 룰 갱신 주기를 변경해 "동적 갱신"의 기여를 분리 측정한다.
#
# --update_interval 1   : 매 에피소드 갱신 (= 제안 모델, 기준)
# --update_interval 5   : 5 에피소드마다 갱신
# --update_interval 0   : 첫 에피소드 후 고정 (이후 갱신 없음)
#
# 사용법:
#   python train_ablation_static.py --seed 42 --update_interval 5
#   python train_ablation_static.py --seed 42 --update_interval 0

import argparse
import matplotlib.pyplot as plt
from utils.logger import Logger
import numpy as np
import torch
import sys
import pandas as pd
import os
import json
import time
import pickle
from datetime import datetime
from collections import deque
from utils.utils import NumpyJSONEncoder
from env.factory_data_generator.data_generator import Generator
from analysis.result_reporter.chart_generator import ChartGenerator
from agents.dqn.rainbow_dqn import RainbowDQN
from env.twin import PlantsimManager
from agents.safety_agent import SimpleSafetyAgent
from agents.llm_agent import LLMStrategicAnalyst

import random as _random

# ── 고정 하이퍼파라미터 ──────────────────────────────────────────
num_episodes       = 100
EVAL_START_EPISODE = 70
num_actions        = 10
chart_frequency    = 10
performance_frequency = 1
iteration_log      = False
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--update_interval', type=int, default=5,
                        help='LLM rule update interval in episodes. '
                             '0=fixed after first episode, 1=every ep, N=every N eps')
    args = parser.parse_args()
    RANDOM_SEED     = args.seed
    UPDATE_INTERVAL = args.update_interval

    if UPDATE_INTERVAL == 0:
        CONDITION = "Ablation_StaticRules_Fixed"
    elif UPDATE_INTERVAL == 1:
        CONDITION = "Ablation_Dynamic_Every1"
    else:
        CONDITION = f"Ablation_StaticRules_Every{UPDATE_INTERVAL}"

    _random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.set_printoptions(suppress=True, precision=6)

    version_no  = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"output/{version_no}_{CONDITION}_s{RANDOM_SEED}"
    log_dir     = "./" + folder_name
    plot_dir    = log_dir + '/plot/'
    for d in [folder_name, folder_name+"/pickle", folder_name+"/csv",
              folder_name+"/log", plot_dir]:
        os.makedirs(d, exist_ok=True)

    logger = Logger(log_dir + '/log/' + version_no + '.log').get_logger()
    logger.info(f'=== {CONDITION} ===')
    logger.info(f'seed={RANDOM_SEED}  update_interval={UPDATE_INTERVAL}  '
                f'episodes={num_episodes}  eval_start={EVAL_START_EPISODE}')

    generator = Generator()
    generator.run()

    llm_analyst  = LLMStrategicAnalyst()
    safety_agent = SimpleSafetyAgent(logger, num_actions=num_actions)
    safety_agent.set_rules([])
    first_rule_set_done = False   # update_interval=0 용 플래그

    dqn = RainbowDQN(num_state=21, num_action=num_actions,
                     num_episodes=EVAL_START_EPISODE, iteration_log=iteration_log)

    plantsim_manager = PlantsimManager()
    plantsim_manager.init(iteration_log, version='23.2', visible='False',
                          license_type='Educational')
    plantsim_manager.initialize_simulation(
        r"E:\SYT\SID_1031\SID_LLM_1031\iplt\IPLT.spp", folder_name, num_episodes)

    batch_size          = 64
    update_target_steps = 300
    iteration_cnt       = 0
    losses              = []
    episode_rewards     = []
    all_episode_results = []
    all_xai_records     = []
    episode_history     = deque(maxlen=5)
    prev_epsilon        = -1

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    for episode in range(num_episodes):
        is_eval = (episode >= EVAL_START_EPISODE)
        episode_start_time = time.time()
        logger.info(f"Episode {episode+1}/{num_episodes} {'[EVAL]' if is_eval else '[TRAIN]'}")
        plantsim_manager.start_simulation(episode)

        total_reward       = 0
        total_target       = 0
        total_urgent_lot   = 0
        total_lot_move     = 0
        total_priority_lot = 0
        episode_action_count = 0
        episode_masked_count = 0

        unique_available_lots_in_episode = set()
        unique_processed_lots_in_episode = set()
        simTime           = 0
        is_first_iteration = True
        wip_log           = []
        last_wip_log_time = 0

        while True:
            if plantsim_manager.get_value("is_done") is True:
                break
            if simTime > simulation_end_time:
                break

            state = plantsim_manager.get_state()
            while plantsim_manager.get_value("WaitingSchedule"):
                simTime = plantsim_manager.get_sim_time()
                if simTime > last_wip_log_time + 3600:
                    wip_log.append(len(plantsim_manager.lot_state))
                    last_wip_log_time = simTime
                if simTime > simulation_end_time:
                    break

                state_summary = plantsim_manager.get_state_summary()
                action_mask, triggered_rules = safety_agent.get_action_mask_with_reason(state_summary)

                action, epsilon, drl_pref, drl_qval, raw_qvals = dqn.select_action(
                    state, episode, action_mask, greedy=is_eval, return_qvals=True)
                if prev_epsilon != epsilon:
                    prev_epsilon = epsilon

                plantsim_manager.execute_action(action, simTime)

                while not plantsim_manager.get_value("WaitingSchedule"):
                    time.sleep(0.1)
                    if plantsim_manager.get_value("is_done"):
                        break
                    if simTime > simulation_end_time:
                        break

                episode_action_count += 1
                if action_mask is not None and not action_mask.all():
                    episode_masked_count += 1

                simTime    = plantsim_manager.get_sim_time()
                next_state = plantsim_manager.get_state()

                reward, target, lot_move, urgent_lot_move, priority_lot_reward, \
                    reward_components, moved_priority_count, \
                    available_lot_ids, processed_lot_ids = \
                    plantsim_manager.calculate_reward(iteration_log)

                total_reward       += reward
                total_target       += target
                total_urgent_lot   += urgent_lot_move
                total_lot_move     += lot_move
                total_priority_lot += priority_lot_reward
                unique_available_lots_in_episode.update(available_lot_ids)
                unique_processed_lots_in_episode.update(processed_lot_ids)

                if not is_eval:
                    if not is_first_iteration:
                        dqn.buffer.add((state, action, reward, next_state))
                    else:
                        is_first_iteration = False

                    if len(dqn.buffer.buffer) > batch_size:
                        loss = dqn.update(batch_size)
                        losses.append(loss)

                    iteration_cnt += 1
                    if iteration_cnt % update_target_steps == 0:
                        dqn.target_net.load_state_dict(dqn.act_net.state_dict())
                else:
                    is_first_iteration = False

                state = next_state
                print_progress_bar(simTime, episode, num_episodes)

        episode_rewards.append(total_reward)

        if (episode + 1) % chart_frequency == 0 or episode > num_episodes - 2:
            chart_generator = ChartGenerator(plot_dir, condition=CONDITION,
                                             eval_start=EVAL_START_EPISODE)
            chart_generator.draw_kpi_trend(all_episode_results, episode)
            chart_generator.draw_reward_convergence(episode_rewards, losses, episode)
            eqp_history = plantsim_manager.get_value("var_eqp_history")
            chart_generator.draw_eqp_chart(eqp_history, episode)
            lot_history = plantsim_manager.get_value("var_lot_history")
            with open(folder_name + "/lot_history.pkl", "wb") as f:
                pickle.dump(lot_history, f)
            chart_generator.draw_lot_chart(lot_history, episode)

        if ((episode + 1) % performance_frequency == 0) or episode == 0 or episode > num_episodes - 2:
            lot_history    = plantsim_manager.get_value("var_lot_history")
            lot_history_df = pd.DataFrame(json.loads(lot_history))
            output_df      = plantsim_manager.get_output_state()
            target_df      = plantsim_manager.get_target_state()
            target_df      = analyze_iplt_and_target(target_df, output_df)
            repair_df      = plantsim_manager.get_repair_qty()

            total_production   = (target_df["qty"] / 20).sum()
            iplt_exceed        = repair_df["count"].sum()
            target_achievement = (target_df["qty"].sum() / target_df["mat_qty"].sum()) * 100
            avg_wip            = np.mean(wip_log) if wip_log else 0

            ulots_df = lot_history_df[lot_history_df['lot_id'].str.startswith('U')].copy()
            if not ulots_df.empty:
                ulot_timing = ulots_df.groupby('lot_id')['sim_time'].agg(['min', 'max']).reset_index()
                avg_ulot_cycle_time = (ulot_timing['max'] - ulot_timing['min']).mean()
            else:
                avg_ulot_cycle_time = 0.0

            total_moved   = len(unique_processed_lots_in_episode)
            total_avail   = len(unique_available_lots_in_episode)
            priority_rate = (total_moved / total_avail * 100) if total_avail > 0 else 0

            episode_summary = {
                'episode':   episode + 1,
                'condition': CONDITION,
                'is_eval':   is_eval,
                'seed':      RANDOM_SEED,
                'update_interval': UPDATE_INTERVAL,
                'AVG_ULOT_CYCLE_TIME': avg_ulot_cycle_time,
                'target_meet_rate':    target_achievement,
                'iplt_over_count':     int(iplt_exceed),
                'total_reward':        total_reward,
                'priority_reward':     total_priority_lot,
                'L2_rules_applied_cumul': llm_analyst.L2_strategy_applied_count,
                'masking_rate': episode_masked_count / max(episode_action_count, 1),
                'total_outputs': float(total_production),
                'priority_process_rate': priority_rate,
                'avg_wip':       avg_wip,
                'total_action_steps': episode_action_count,
            }
            all_episode_results.append(episode_summary)

        pd.DataFrame(all_episode_results).to_csv(
            folder_name + f"/csv/results_{CONDITION}.csv", index=False)
        if (episode + 1) % 10 == 0 and not is_eval:
            dqn.save_checkpoint(folder_name + f"/dqn_ckpt_ep{episode+1}.pt")

        # train_proposed.py와 동일한 순서: 먼저 history에 추가 → LLM에 현재 ep 포함
        episode_history.append(episode_summary)

        # ── 갱신 주기 로직 ──────────────────────────────────────
        # update_interval=0: 첫 번째 ep 이후 갱신 없음 (룰 고정)
        # update_interval=N: N 에피소드마다 갱신
        if not is_eval:
            should_update = False
            if UPDATE_INTERVAL == 0:
                if not first_rule_set_done:
                    should_update = True
                    first_rule_set_done = True
            else:
                # episode는 0-indexed → (episode+1) % N == 0
                should_update = ((episode + 1) % UPDATE_INTERVAL == 0)

            if should_update:
                try:
                    new_rules, xai_rec = llm_analyst.analyze_and_update_rules(
                        episode_results=episode_summary,
                        previous_rules=safety_agent.current_rules,
                        episode_history=list(episode_history)
                    )
                    logger.info(
                        f"[L2] ep={episode+1} rules updated "
                        f"(interval={UPDATE_INTERVAL}): {len(new_rules)} rules")
                    safety_agent.set_rules(new_rules)
                    all_xai_records.append(xai_rec)
                except Exception as e:
                    logger.error(f"[L2] Failed to update rules: {e}", exc_info=True)
            else:
                logger.info(
                    f"[L2] ep={episode+1} rules NOT updated "
                    f"(interval={UPDATE_INTERVAL}, holding current {len(safety_agent.current_rules)} rules)")

    results_df = pd.DataFrame(all_episode_results)
    results_df.to_csv(folder_name + f"/csv/rewards_data.csv", index=False)
    results_df.to_csv(folder_name + f"/csv/results_{CONDITION}.csv", index=False)
    with open(folder_name + "/csv/xai_decisions.json", "w", encoding="utf-8") as f:
        json.dump(all_xai_records, f, ensure_ascii=False, indent=2, default=str)

    plantsim_manager.quit()
    print(f"\n[DONE] {CONDITION} seed={RANDOM_SEED} → {folder_name}")
