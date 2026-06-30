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
from dqn.rainbow_dqn import RainbowDQN
from twin import PlantsimManager

#####################
#   (A) Pure DRL    #
#####################
num_episodes = 100
EVAL_START_EPISODE = 70   # 에피소드 71~100: 평가 전용 (학습 없음, greedy 정책)
RANDOM_SEED = 42
num_actions = 10
chart_frequency = 10
performance_frequency = 1
iteration_log = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
simulation_end_time = 86400


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


def format_log(array):
    split_sizes = [1, 4, 4, 8, 4]
    formatted_values = [f"{x:.4f}" for x in array]
    output_lines = []
    start = 0
    for size in split_sizes:
        end = start + size
        slice_data = formatted_values[start:end]
        if slice_data:
            output_lines.append(", ".join(slice_data))
        start = end
    return "\n".join(output_lines)


def get_action(state, prev_epsilon, greedy=False):
    action, epsilon = dqn.select_action(state, episode, greedy=greedy)
    if prev_epsilon != epsilon:
        print('epsilon : ' + str(epsilon))
    if iteration_log:
        logger.info('Decide and Execute Action action : ' + str(action))
    return action, epsilon


def calculate_avg_processing_time(lot_history_df, target_process_id):
    target_df = lot_history_df[
        (lot_history_df['lot_id'].str.startswith('U')) &
        (lot_history_df['process_id'] == target_process_id)
    ].copy()
    if target_df.empty:
        return np.nan
    timing_df = target_df.groupby('lot_id')['sim_time'].agg(['min', 'max']).reset_index()
    timing_df['proc_time'] = timing_df['max'] - timing_df['min']
    return timing_df['proc_time'].mean()


if __name__ == '__main__':
    import argparse, random as _random
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    RANDOM_SEED = args.seed

    _random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.set_printoptions(suppress=True, precision=6)

    version_no = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"output/{version_no}_PureDRL_s{RANDOM_SEED}"
    log_dir = "./" + folder_name
    plot_dir = log_dir + '/plot/'
    for d in [folder_name, folder_name+"/pickle", folder_name+"/csv", folder_name+"/log", plot_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    logger = Logger(log_dir + '/log/' + version_no + '.log').get_logger()
    logger.info('=== (A) Pure DRL Baseline ===')
    logger.info(f'Episodes: {num_episodes}  Train: 1~{EVAL_START_EPISODE}  Eval: {EVAL_START_EPISODE+1}~{num_episodes}')
    logger.info(f'Random seed: {RANDOM_SEED}')

    logger.info("Data Generator Run Start")
    generator = Generator()
    generator.run()
    logger.info("Data Generator Run Complete")

    batch_size = 64
    update_target_steps = 300

    episode_rewards = []
    production_rewards = []
    risk_reward = []
    lot_processing_rewards = []
    priority_rewards = []
    all_episode_results = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    # DQN: num_episodes=EVAL_START_EPISODE → epsilon이 훈련 종료 시 최솟값 도달
    dqn = RainbowDQN(num_state=21, num_action=num_actions,
                     num_episodes=EVAL_START_EPISODE, iteration_log=iteration_log)
    iteration_cnt = 0

    plantsim_manager = PlantsimManager()
    plantsim_manager.init(iteration_log, version='23.2', visible='False', license_type='Educational')
    plantsim_manager.initialize_simulation(
        r"E:\SYT\SID_1031\SID_LLM_1031\iplt\IPLT.spp", folder_name, num_episodes)

    losses = []
    total_get_state_elapsed_time = 0
    total_get_action_elapsed_time = 0
    total_execute_action_elapsed_time = 0
    total_lot_execute_time = 0
    total_eqp_execute_time = 0
    total_calculate_reward_action_elapsed_time = 0
    total_update_network_elapsed_time = 0
    total_summary_elapsed_time = 0
    prev_epsilon = -1

    for episode in range(num_episodes):
        is_eval = (episode >= EVAL_START_EPISODE)
        episode_start_time = time.time()
        logger.info(f"Episode {episode + 1}/{num_episodes} {'[EVAL]' if is_eval else '[TRAIN]'}")
        plantsim_manager.start_simulation(episode)

        total_reward = 0
        total_target = 0
        total_urgent_lot = 0
        total_lot_move = 0
        total_priority_lot = 0
        unique_available_lots_in_episode = set()
        unique_processed_lots_in_episode = set()
        simTime = 0
        is_first_iteration = True
        wip_log = []
        last_wip_log_time = 0

        while True:
            if plantsim_manager.get_value("is_done") is True:
                break
            if simTime > simulation_end_time:
                break

            state = plantsim_manager.get_state()
            while plantsim_manager.get_value("WaitingSchedule"):
                if iteration_log:
                    logger.info('')
                    logger.info('###############################')

                simTime = plantsim_manager.get_sim_time()

                if simTime > last_wip_log_time + 3600:
                    wip_log.append(len(plantsim_manager.lot_state))
                    last_wip_log_time = simTime

                if simTime > simulation_end_time:
                    break

                # Pure DRL: 마스킹 없음, LLM 없음
                get_action_start_time = time.time()
                action, epsilon = get_action(state, prev_epsilon, greedy=is_eval)
                prev_epsilon = epsilon
                total_get_action_elapsed_time += time.time() - get_action_start_time

                execute_action_start_time = time.time()
                lot_elapsed_time, eqp_elapsed_time = plantsim_manager.execute_action(action, simTime)
                total_lot_execute_time += lot_elapsed_time
                total_eqp_execute_time += eqp_elapsed_time
                total_execute_action_elapsed_time += time.time() - execute_action_start_time

                while not plantsim_manager.get_value("WaitingSchedule"):
                    time.sleep(0.1)
                    if plantsim_manager.get_value("is_done") is True:
                        break
                    if simTime > simulation_end_time:
                        break

                get_state_start_time = time.time()
                simTime = plantsim_manager.get_sim_time()
                if iteration_log:
                    logger.info("simTime : " + str(simTime))

                next_state = plantsim_manager.get_state()
                total_get_state_elapsed_time += time.time() - get_state_start_time

                if iteration_log:
                    logger.info('state : ' + format_log(state))
                    logger.info('action : ' + str(action))

                calculate_reward_start_time = time.time()
                reward, target, lot_move, urgent_lot_move, priority_lot_reward, reward_components, \
                    moved_priority_count, available_lot_ids, processed_lot_ids = \
                    plantsim_manager.calculate_reward(iteration_log)
                total_calculate_reward_action_elapsed_time += time.time() - calculate_reward_start_time

                update_network_start_time = time.time()
                total_reward += reward
                total_target += target
                total_urgent_lot += urgent_lot_move
                total_lot_move += lot_move
                total_priority_lot += priority_lot_reward
                unique_available_lots_in_episode.update(available_lot_ids)
                unique_processed_lots_in_episode.update(processed_lot_ids)

                if iteration_log:
                    xrl_data = {"simTime": simTime, "action": action,
                                "reward_components": reward_components}
                    logger.info(f"[XRL_DATA] {json.dumps(xrl_data, cls=NumpyJSONEncoder)}")

                # 훈련/평가 분기
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
                total_update_network_elapsed_time += time.time() - update_network_start_time
                print_progress_bar(simTime, episode, num_episodes)

        summary_start_time = time.time()
        episode_rewards.append(total_reward)
        production_rewards.append(total_target)
        risk_reward.append(total_urgent_lot)
        lot_processing_rewards.append(total_lot_move)
        priority_rewards.append(total_priority_lot)

        if (episode + 1) % chart_frequency == 0 or (episode > num_episodes - 2):
            chart_generator = ChartGenerator(plot_dir, condition='PureDRL',
                                             eval_start=EVAL_START_EPISODE)
            chart_generator.draw_kpi_trend(all_episode_results, episode)
            chart_generator.draw_reward_convergence(episode_rewards, losses, episode)
            eqp_history = plantsim_manager.get_value("var_eqp_history")
            chart_generator.draw_eqp_chart(eqp_history, episode)
            lot_history = plantsim_manager.get_value("var_lot_history")
            with open(folder_name + "/lot_history.pkl", "wb") as f:
                pickle.dump(lot_history, f)
            chart_generator.draw_lot_chart(lot_history, episode)

        if ((episode + 1) % performance_frequency == 0) or (episode == 0) or (episode > num_episodes - 2):
            lot_history = plantsim_manager.get_value("var_lot_history")
            lot_history_df = pd.DataFrame(json.loads(lot_history))
            output_df = plantsim_manager.get_output_state()
            target_df = plantsim_manager.get_target_state()
            target_df = analyze_iplt_and_target(target_df, output_df)
            repair_df = plantsim_manager.get_repair_qty()

            total_production = (target_df["qty"] / 20).sum()
            iplt_exceed = repair_df["count"].sum()
            satisfied_devices = target_df[target_df["device_id"].isin(repair_df["device_id"])]
            iplt_satisfied = (satisfied_devices["qty"] / 20).sum()
            target_achievement = (target_df["qty"].sum() / target_df["mat_qty"].sum()) * 100
            avg_wip = np.mean(wip_log) if wip_log else 0

            ulots_df = lot_history_df[lot_history_df['lot_id'].str.startswith('U')].copy()
            if not ulots_df.empty:
                ulot_timing = ulots_df.groupby('lot_id')['sim_time'].agg(['min', 'max']).reset_index()
                avg_ulot_cycle_time = (ulot_timing['max'] - ulot_timing['min']).mean()
            else:
                avg_ulot_cycle_time = 0.0

            result = {
                "Total Outputs": float(total_production),
                "IPLT Over": int(iplt_exceed),
                "IPLT Meet": float(iplt_satisfied),
                "Target Meet": float(target_achievement)
            }
            logger.info(result)

            total_moved_priority_lots = len(unique_processed_lots_in_episode)
            total_available_priority_lots = len(unique_available_lots_in_episode)
            priority_rate = (total_moved_priority_lots / total_available_priority_lots * 100
                             if total_available_priority_lots > 0 else 0)

            episode_summary = {
                'episode':   episode + 1,
                'condition': 'PureDRL',
                'is_eval':   is_eval,
                'seed':      RANDOM_SEED,
                'AVG_ULOT_CYCLE_TIME': avg_ulot_cycle_time,
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

        total_summary_elapsed_time += time.time() - summary_start_time

        elapsed_time = time.time() - episode_start_time
        if iteration_log:
            logger.info(
                f'episode elapsed: {elapsed_time:.1f}s  reward: {total_reward:.2f}  '
                f'target: {total_target:.2f}  lot_move: {total_lot_move:.2f}')

    results_df = pd.DataFrame(all_episode_results)
    results_df.to_csv(folder_name + "/csv/rewards_data.csv", index=False)
    results_df.to_csv(folder_name + "/csv/results_PureDRL.csv", index=False)

    logger.info(f"get state     : {total_get_state_elapsed_time:.4f}")
    logger.info(f"get action    : {total_get_action_elapsed_time:.4f}")
    logger.info(f"execute action: {total_execute_action_elapsed_time:.4f}")
    logger.info(f"calc reward   : {total_calculate_reward_action_elapsed_time:.4f}")
    logger.info(f"update network: {total_update_network_elapsed_time:.4f}")
    logger.info(f"summary       : {total_summary_elapsed_time:.4f}")

    plantsim_manager.quit()
