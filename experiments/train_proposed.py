import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

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


#####################
#     Init Value    #0
#####################
num_episodes = 100
EVAL_START_EPISODE = 70
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

    target_df["achievement_rate"] = np.where(target_df["mat_qty"] > 0, (target_df["qty"] / target_df["mat_qty"]) * 100,0)
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


def get_action(state, prev_epsilon, mask=None, greedy=False):
    if iteration_log and mask is not None:
        masked_indices = np.where(~mask)[0]
        if len(masked_indices) > 0:
            logger.info(f'[L1 MASKED] Actions {masked_indices} forbidden by LLM rule.')
        else:
            logger.info('[L1 MASKED] No actions currently forbidden (Free policy).')

    action, epsilon = dqn.select_action(state, episode, mask, greedy=greedy)

    if prev_epsilon != epsilon:
        print('epsilon : ' + str(epsilon))
    if iteration_log:
        logger.info('Decide and Execute Action action : ' + str(action))
    return action, epsilon


def calculate_avg_processing_time(lot_history_df, target_process_id):
    """Mean pure processing time for a given Process ID."""







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
    parser.add_argument('--urgent_entry_type', type=str, default='early',
                        choices=['early', 'mid', 'late', 'mixed'],
                        help='긴급 Lot 진입 유형: early(투입형), mid(전환형), late(막바지형), mixed(혼합형)')
    args = parser.parse_args()
    RANDOM_SEED = args.seed
    URGENT_ENTRY_TYPE = args.urgent_entry_type

    model_type = "Proposed_FullLLM"
    np.set_printoptions(suppress=True, precision=6)
    version_no  = datetime.now().strftime('%Y%m%d_%H%M%S')
    entry_suffix = f"_{URGENT_ENTRY_TYPE}" if URGENT_ENTRY_TYPE != "early" else ""
    folder_name = f"output/{version_no}_{model_type}{entry_suffix}_s{RANDOM_SEED}"
    log_dir     = "./" + folder_name
    plot_dir    = log_dir + '/plot/'
    for d in [folder_name, folder_name+"/pickle", folder_name+"/csv",
              folder_name+"/log", plot_dir]:
        os.makedirs(d, exist_ok=True)

    _random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    logger = Logger(log_dir +'/log/' + version_no + '.log').get_logger()
    logger.info('Episode : ' + str(num_episodes))
    logger.info(f'Train episodes: 1~{EVAL_START_EPISODE}, Eval episodes: {EVAL_START_EPISODE+1}~{num_episodes}')
    logger.info(f'Random seed: {RANDOM_SEED}')
    logger.info('Chart Verbose : ' + str(chart_frequency))
    logger.info('Performance Verbose : ' + str(performance_frequency))
    logger.info('Write Iteration Log : ' + str(iteration_log))
    # logger.info(torch.cuda.get_device_name(0))
    # logger.info(torch.cuda.is_available())
    logger.info('chart_frequency : ' + str(chart_frequency))

    logger.info("Start System")
    logger.info("Data Generator Run Start")
    generator = Generator()
    generator.run(urgent_entry_type=URGENT_ENTRY_TYPE)
    logger.info("Data Generator Run Complete")


    try:

        llm_analyst = LLMStrategicAnalyst()

        safety_agent = SimpleSafetyAgent(logger, num_actions=num_actions)


        logger.info("Setting initial safety rules (empty list)...")
        safety_agent.set_rules([])

        logger.info("L1/L2 Agents Initialized Successfully.")

    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        sys.exit(1)


    batch_size = 64
    update_target_steps = 300

    episode_rewards = []
    production_rewards = []
    risk_reward = []
    lot_processing_rewards = []
    priority_rewards = []
    all_episode_results = []
    all_xai_records = []
    all_xai_step_records = []
    episode_history = deque(maxlen=5)

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    dqn = RainbowDQN(num_state=21, num_action=num_actions, num_episodes=EVAL_START_EPISODE, iteration_log=iteration_log)
    iteration_cnt = 0

    plantsim_manager = PlantsimManager()
    plantsim_manager.init(iteration_log, version='23.2', visible='False', license_type='Educational')
    plantsim_manager.initialize_simulation(r"E:\SYT\SID_1031\SID_LLM_1031\iplt\IPLT.spp", folder_name, num_episodes)

    decided_action = 1
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
    L1_strategy_switch_count = 0
    L1_total_action_steps = 0


    for episode in range(num_episodes):
        episode_start_time = time.time()
        logger.info(f"Episode {episode + 1}/{num_episodes} " + " action : " + str(episode))
        plantsim_manager.start_simulation(episode)
        episode_summary = {}
        total_reward = 0
        total_target = 0
        total_urgent_lot = 0
        total_lot_move = 0
        total_priority_lot = 0

        episode_action_count = 0
        episode_masked_count = 0
        last_action = -1

        unique_available_lots_in_episode = set()
        unique_processed_lots_in_episode = set()

        is_eval = (episode >= EVAL_START_EPISODE)
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
                    logger.info('###############################')

                simTime = plantsim_manager.get_sim_time()



                if simTime > last_wip_log_time + 3600:

                    current_wip = len(plantsim_manager.lot_state)
                    wip_log.append(current_wip)
                    last_wip_log_time = simTime


                if simTime > simulation_end_time:
                    break


                state_summary = plantsim_manager.get_state_summary()
                action_mask, triggered_rules = safety_agent.get_action_mask_with_reason(state_summary)


                get_action_start_time = time.time()
                action, epsilon, drl_pref, drl_qval, raw_qvals = dqn.select_action(
                    state, episode, action_mask, greedy=is_eval, return_qvals=True)
                ACTION_SOURCE = 'RL'
                prev_epsilon = epsilon


                ##################################
                #         Execute Action         #
                ##################################
                execute_action_start_time = time.time()


                lot_elapsed_time, eqp_elapsed_time = plantsim_manager.execute_action(
                    action, simTime)

                total_lot_execute_time = total_lot_execute_time + lot_elapsed_time
                total_eqp_execute_time = total_eqp_execute_time + eqp_elapsed_time
                execute_action_end_time = time.time()
                execute_action_elapsed_time = execute_action_end_time - execute_action_start_time
                total_execute_action_elapsed_time += execute_action_elapsed_time

                ##################################
                #            Waiting             #
                ##################################
                while not plantsim_manager.get_value("WaitingSchedule"):
                    time.sleep(0.1)  #
                    if plantsim_manager.get_value("is_done") is True:
                        break
                    if simTime > simulation_end_time:
                        break

                get_state_start_time = time.time()
                current_action = action
                episode_action_count += 1
                llm_intervened = (action_mask is not None and not action_mask.all())
                if llm_intervened:
                    episode_masked_count += 1

                    all_xai_step_records.append({
                        "episode":       episode + 1,
                        "simTime":       simTime,
                        "is_eval":       is_eval,

                        "drl_preferred_action": drl_pref,
                        "drl_preferred_qval":   round(float(drl_qval), 4),
                        "drl_top3": [(int(i), round(float(raw_qvals[i]), 4))
                                     for i in np.argsort(raw_qvals)[::-1][:3]],

                        "final_action":   action,
                        "llm_overrode":   (drl_pref != action),
                        "triggered_rules": triggered_rules,

                        "state_summary":  state_summary,
                    })
                last_action = current_action


                simTime = plantsim_manager.get_sim_time()
                if iteration_log:
                    logger.info("simTime : " + str(simTime))
                    logger.info("================")
                    logger.info("get next state")

                ##################################
                #          Get Next State        #
                ##################################
                next_state = plantsim_manager.get_state()
                get_state_end_time = time.time()
                get_state_elapsed_time = get_state_end_time - get_state_start_time
                total_get_state_elapsed_time += get_state_elapsed_time
                if iteration_log:
                    logger.info('state : ' + format_log(state))
                    logger.info('action : ' + str(action))
                    logger.info(f'iplt\n{plantsim_manager.iplt_state}')
                    logger.info(f'output\n{plantsim_manager.output_state}')
                    logger.info(f'target\n{plantsim_manager.target_state}')

                ##################################
                #        Calculate Reward        #
                ##################################
                calculate_reward_start_time = time.time()


                reward, target, lot_move, urgent_lot_move, priority_lot_reward, reward_components, \
                    moved_priority_count, available_lot_ids, processed_lot_ids = \
                    plantsim_manager.calculate_reward(iteration_log)

                calculate_reward_end_time = time.time()
                calculate_reward_elapsed_time = calculate_reward_end_time - calculate_reward_start_time
                total_calculate_reward_action_elapsed_time += calculate_reward_elapsed_time
                update_network_start_time = time.time()

                total_reward += reward
                total_target += target
                total_urgent_lot += urgent_lot_move
                total_lot_move += lot_move
                total_priority_lot += priority_lot_reward


                unique_available_lots_in_episode.update(available_lot_ids)
                unique_processed_lots_in_episode.update(processed_lot_ids)

                if iteration_log:

                    xrl_data = {
                        "simTime": simTime,
                        "action": action,
                        "source": ACTION_SOURCE,
                        "reward_components": reward_components
                    }
                    logger.info(f"[XRL_DATA] {json.dumps(xrl_data, cls=NumpyJSONEncoder)}")

                if not is_eval:
                    if not is_first_iteration:
                        dqn.buffer.add((state, action, reward, next_state))
                    else:
                        is_first_iteration = False

                    if len(dqn.buffer.buffer) > batch_size:
                        loss = dqn.update(batch_size)
                        losses.append(loss)

                    iteration_cnt = iteration_cnt + 1
                    if iteration_cnt % update_target_steps == 0:
                        dqn.target_net.load_state_dict(dqn.act_net.state_dict())
                else:
                    is_first_iteration = False

                state = next_state

                update_network_end_time = time.time()
                update_network_elapsed_time = update_network_end_time - update_network_start_time
                total_update_network_elapsed_time += update_network_elapsed_time

                print_progress_bar(simTime, episode, num_episodes)

        summary_start_time = time.time()
        episode_rewards.append(total_reward)
        production_rewards.append(total_target)
        risk_reward.append(total_urgent_lot)
        lot_processing_rewards.append(total_lot_move)

        priority_rewards.append(total_priority_lot)


        if (episode + 1) % chart_frequency == 0 or (episode > num_episodes - 2):
            chart_generator = ChartGenerator(plot_dir, condition=model_type,
                                             eval_start=EVAL_START_EPISODE)

            chart_generator.draw_kpi_trend(all_episode_results, episode)

            chart_generator.draw_reward_convergence(episode_rewards, losses, episode)

            chart_generator.draw_governance_activity(all_episode_results, episode)

            eqp_history = plantsim_manager.get_value("var_eqp_history")
            chart_generator.draw_eqp_chart(eqp_history, episode)

            lot_history = plantsim_manager.get_value("var_lot_history")

            with open(folder_name + "/lot_history.pkl", "wb") as f:
                pickle.dump(lot_history, f)
            chart_generator.draw_lot_chart(lot_history, episode)

        if ((episode + 1) % performance_frequency == 0) or (episode == 0) or (episode > num_episodes - 2):
            lot_history = plantsim_manager.get_value("var_lot_history")
            fixed_string = json.loads(lot_history)
            lot_history_df = pd.DataFrame(fixed_string)
            iplt_df = plantsim_manager.get_iplt_state()
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
            # === U-Lot Cycle Time Calculation ===

            ulots_df = lot_history_df[lot_history_df['lot_id'].str.startswith('U')].copy()

            if not ulots_df.empty:


                ulot_timing = ulots_df.groupby('lot_id')['sim_time'].agg(['min', 'max']).reset_index()


                avg_ulot_cycle_time = (ulot_timing['max'] - ulot_timing['min']).mean()
            else:
                avg_ulot_cycle_time = 0.0


            AVG_P4_PROC_TIME = calculate_avg_processing_time(lot_history_df, target_process_id=4)
            AVG_P5_PROC_TIME = calculate_avg_processing_time(lot_history_df, target_process_id=5)


            if not np.isnan(AVG_P4_PROC_TIME) and not np.isnan(AVG_P5_PROC_TIME):
                T_DELTA_P5_P4 = AVG_P5_PROC_TIME - AVG_P4_PROC_TIME
            else:
                T_DELTA_P5_P4 = np.nan
            result = {
                "Total Outputs": float(total_production),
                "IPLT Over": int(iplt_exceed),
                "IPLT Meet": float(iplt_satisfied),
                "Target Meet": float(target_achievement)
            }

            logger.info(result)

            total_moved_priority_lots = len(unique_processed_lots_in_episode)
            total_available_priority_lots = len(unique_available_lots_in_episode)

            priority_rate = (total_moved_priority_lots / total_available_priority_lots) * 100 if total_available_priority_lots > 0 else 0

            episode_summary = {

                'episode':   episode + 1,
                'condition': model_type,
                'is_eval':   is_eval,
                'seed':      RANDOM_SEED,

                'AVG_ULOT_CYCLE_TIME': avg_ulot_cycle_time,
                'target_meet_rate':    target_achievement,
                'iplt_over_count':     int(iplt_exceed),

                'total_reward':    total_reward,
                'priority_reward': total_priority_lot,

                'L2_rules_applied_cumul': llm_analyst.L2_strategy_applied_count,
                'masking_rate': episode_masked_count / max(episode_action_count, 1),

                'total_outputs':       float(total_production),
                'priority_process_rate': priority_rate,
                'avg_wip':             avg_wip,
                'total_action_steps':  episode_action_count,
            }

            all_episode_results.append(episode_summary)



        _interim_df = pd.DataFrame(all_episode_results)
        _interim_df.to_csv(folder_name + f"/csv/results_{model_type}.csv", index=False)

        if (episode + 1) % 10 == 0 and not is_eval:
            dqn.save_checkpoint(folder_name + f"/dqn_ckpt_ep{episode+1}.pt")
            logger.info(f"[CKPT] DQN checkpoint saved at episode {episode+1}")
        # ──────────────────────────────────────────────────────────

        summary_end_time = time.time()
        summary_elapsed_time = summary_end_time - summary_start_time
        total_summary_elapsed_time += summary_elapsed_time

        episode_end_time = time.time()  #
        elapsed_time = episode_end_time - episode_start_time

        if iteration_log:
            logger.info(
                'episode elapsed time : ' + str(elapsed_time) + ' total reward : ' + str(total_reward) + ' production : ' + str(total_target) + ' exceed : ' + str(total_urgent_lot) + ' lot_move : ' + str(total_lot_move))


        episode_history.append(episode_summary)
        if not is_eval:
            logger.info(f"[L2] Episode {episode + 1} finished. Requesting strategic analysis...")
            try:
                new_rules, xai_rec = llm_analyst.analyze_and_update_rules(
                    episode_results=episode_summary,
                    previous_rules=safety_agent.current_rules,
                    episode_history=list(episode_history)
                )
                logger.info(f"[L2] Received {len(new_rules)} rules. Reasoning: {xai_rec.get('reasoning','')[:80]}")
                safety_agent.set_rules(new_rules)
                all_xai_records.append(xai_rec)
            except Exception as e:
                logger.error(f"[L2] Failed to update strategy.", exc_info=True)
                logger.info("Using previous settings.")
        else:
            logger.info(f"[EVAL] Episode {episode + 1} is evaluation mode. L2 rules frozen.")

    # df = pd.DataFrame({
    #     "episode_rewards": episode_rewards,
    #     "production_rewards": production_rewards,
    #     "risk_reward": risk_reward,
    #     "lot_processing_rewards": lot_processing_rewards
    # })
    #

    # df.to_csv(folder_name + f"/csv/rewards_data.csv", index=False)

    results_df = pd.DataFrame(all_episode_results)



    results_df.to_csv(folder_name + f"/csv/rewards_data.csv", index=False)

    import json as _json

    xai_path = folder_name + "/csv/xai_decisions.json"
    with open(xai_path, "w", encoding="utf-8") as _f:
        _json.dump(all_xai_records, _f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"XAI episode decisions saved: {xai_path} ({len(all_xai_records)} records)")

    xai_step_path = folder_name + "/csv/xai_step_interventions.json"
    with open(xai_step_path, "w", encoding="utf-8") as _f:
        _json.dump(all_xai_step_records, _f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"XAI step interventions saved: {xai_step_path} ({len(all_xai_step_records)} records)")
    results_df.to_csv(folder_name + f"/csv/results_{model_type}.csv", index=False)

    logger.info(f"get state : {total_get_state_elapsed_time:.4f}")
    logger.info(f"get action : {total_get_action_elapsed_time:.4f}")
    logger.info(f"execute action : {total_execute_action_elapsed_time:.4f}")
    logger.info(f"     lot_execute_time : {total_lot_execute_time:.4f}")
    logger.info(f"     eqp_execute_time : {total_eqp_execute_time:.4f}")
    logger.info(f"calculate reward : {total_calculate_reward_action_elapsed_time:.4f}")
    logger.info(f"update_network : {total_update_network_elapsed_time:.4f}")
    logger.info(f"summary : {total_summary_elapsed_time:.4f}")

    plantsim_manager.quit()
