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
# ✅ (신규) L1, L2 에이전트 import
from simple_safety_agent import SimpleSafetyAgent
from llm_agent import LLMStrategicAnalyst # (이름이 LLMStrategicAnalyst로 변경됨)


#####################
#     Init Value    #0
#####################
num_episodes = 50
num_actions = 10 # ✅ (신규) 액션 개수 변수 (DQN/L1과 일치)
chart_frequency = 10
performance_frequency = 1
iteration_log = True

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
simulation_end_time = 86400

def print_progress_bar(simTime, episode, total_episode):
    bar_length = 50  # 진행 바 길이
    progress = simTime / (simulation_end_time - 2000)  # 진행률 계산
    filled_length = int(bar_length * progress)  # 현재 진행된 길이
    bar = "█" * filled_length + "-" * (bar_length - filled_length)  # 진행 바 그리기
    sys.stdout.write(f"\repisode {episode + 1}/{total_episode}: |{bar}|")  # 한 줄에 출력
    sys.stdout.flush()  # 버퍼 비우기 (즉시 반영)

def analyze_iplt_and_target(target_df, output_df):
    output_qty_map = output_df.set_index("device_id")["qty"].to_dict()
    target_df["qty"] = target_df["device_id"].map(output_qty_map).fillna(0)

    target_df["achievement_rate"] = np.where(target_df["mat_qty"] > 0, (target_df["qty"] / target_df["mat_qty"]) * 100,0)
    return target_df

def format_log(array):
    # ✅ 출력 개수 설정
    split_sizes = [1, 4, 4, 8, 4]
    formatted_values = [f"{x:.4f}" for x in array]  # 소수점 4자리 포맷
    output_lines = []

    start = 0
    for size in split_sizes:
        end = start + size
        slice_data = formatted_values[start:end]
        if slice_data:
            output_lines.append(", ".join(slice_data))
        start = end  # 다음 구간으로 이동

    return "\n".join(output_lines)  # 줄바꿈으로 조합


def get_action(state, prev_epsilon, mask=None):
    # 🚨 (수정) mask를 dqn.select_action으로 전달
    if iteration_log and mask is not None:
        masked_indices = np.where(~mask)[0]
        if len(masked_indices) > 0:
            logger.info(f'[L1 MASKED] Actions {masked_indices} forbidden by LLM rule.')
        else:
            logger.info('[L1 MASKED] No actions currently forbidden (Free policy).')

    action, epsilon = dqn.select_action(state, episode, mask)

    if prev_epsilon != epsilon:
        print('epsilon : ' + str(epsilon))
    if iteration_log:
        logger.info('Decide and Execute Action action : ' + str(action))
    return action, epsilon


def calculate_avg_processing_time(lot_history_df, target_process_id):
    """특정 Process ID의 순수 처리 시간(Processing Time) 평균을 계산합니다."""

    # 🚨 Lot History의 컬럼 이름을 가정합니다: lot_id, process_id, sim_time, state

    # Process의 PROCESSING 시작/완료 이벤트 필터링
    # Lot이 'PROCESSING' 상태에 들어간 후 해당 Process ID를 떠나는 시점을 찾아야 합니다.

    # 1. 대상 Process 및 Lot ID 필터링 (ULot만 대상)
    target_df = lot_history_df[
        (lot_history_df['lot_id'].str.startswith('U')) &
        (lot_history_df['process_id'] == target_process_id)
        ].copy()

    if target_df.empty:
        return np.nan

    # 2. Lot ID별 최초 시점 (Processing Start) 및 최종 시점 (Completion) 찾기
    # Process 내에서 가장 이른 시간 = Process Start (Processing 상태 진입 근사치)
    # Process 내에서 가장 늦은 시간 = Process End (Completion 근사치)

    timing_df = target_df.groupby('lot_id')['sim_time'].agg(['min', 'max']).reset_index()

    # 순수 처리 시간 (Processing Time) = max_time - min_time
    timing_df['proc_time'] = timing_df['max'] - timing_df['min']

    # Process 4와 Process 5의 경우, Lot이 Process 4/5 버퍼에 도착한 후의 모든 이벤트를 포함하므로,
    # 이 'max - min' 값은 해당 Process에서의 총 소요 시간(Waiting + Processing)에 근사합니다.

    # Process 4와 5의 'Processing' 이벤트만 명확하게 필터링하여 순수 처리 시간을 측정하려면,
    # 해당 Lot ID와 다음 Lot ID가 Process ID를 떠날 때의 시간을 찾아야 하지만,
    # 현재 데이터 구조에서는 min/max를 사용하는 것이 가장 안전한 근사치입니다.

    return timing_df['proc_time'].mean()

if __name__ == '__main__':
    model_type = "Baseline_C"
    strategy_count = 1

    # 전문가가 정의한 "합리적" 고정 규칙 설정
    # 긴급 물량이 5개 이상일 때만 액션을 제한하도록 설정 (짚인형 비판 회피)
    from llm_agent import SafetyRule

    static_expert_rule = SafetyRule(
        metric="urgent_priority_lot_count",
        operator=">=",
        threshold=5,
        forbidden_actions=[0, 1, 2, 3, 4, 5]  # 생산성 위주 액션 차단
    )



    np.set_printoptions(suppress=True, precision=6)
    version_no = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"output/{version_no}"
    log_dir = "./" + folder_name
    plot_dir = log_dir + '/plot/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if not os.path.exists(folder_name + f"/pickle"):
        os.makedirs(folder_name + f"/pickle")
    if not os.path.exists(folder_name + f"/csv"):
        os.makedirs(folder_name + f"/csv")
    if not os.path.exists(folder_name + f"/log"):
        os.makedirs(folder_name + f"/log")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    logger = Logger(log_dir +'/log/' + version_no + '.log').get_logger()
    logger.info('Episode : ' + str(num_episodes))
    logger.info('Chart Verbose : ' + str(chart_frequency))
    logger.info('Performance Verbose : ' + str(performance_frequency))
    logger.info('Write Iteration Log : ' + str(iteration_log))
    # logger.info(torch.cuda.get_device_name(0))
    # logger.info(torch.cuda.is_available())
    logger.info('chart_frequency : ' + str(chart_frequency))

    logger.info("Start System")
    logger.info("Data Generator Run Start")
    generator = Generator()
    generator.run()
    logger.info("Data Generator Run Complete")

    # 🚨 (수정) L1, L2 에이전트 초기화
    try:
         # L2 전략 분석가 (느림, 에피소드 종료 후 사용)
        llm_analyst = LLMStrategicAnalyst()
         # ✅ (수정) L1 에이전트에 num_actions 전달
        safety_agent = SimpleSafetyAgent(logger, num_actions=num_actions)

        safety_agent.set_rules([static_expert_rule])
        logger.info(f"[Baseline C] Static Expert Rule Initialized: {static_expert_rule}")

         # ✅ [수정] 'Initial Run' 시 LLM을 호출하지 않고, L1을 빈 룰로 초기화
        # logger.info("Setting initial safety rules (empty list)...")
        # safety_agent.set_rules([])  # ⬅️ 빈 룰 리스트로 직접 설정

        logger.info("L1/L2 Agents Initialized Successfully.")

    except Exception as e:
        logger.error(f"Agent 초기화 실패: {e}")
        sys.exit(1)

    # ✅ 학습 하이퍼파라미터
    batch_size = 64
    update_target_steps = 300  # Target 네트워크 업데이트 주기

    episode_rewards = []
    production_rewards = []
    risk_reward = []
    lot_processing_rewards = []
    priority_rewards = []
    all_episode_results = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    dqn = RainbowDQN(num_state=21, num_action=num_actions, num_episodes=num_episodes, iteration_log=iteration_log)
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
        total_reward = 0
        total_target = 0
        total_urgent_lot = 0
        total_lot_move = 0
        total_priority_lot = 0 # ✅ (신규) 긴급 물량 보상 합계 변수
        # 🚨 [추가] 에피소드 별 액션 스텝 카운터 및 마스킹 카운터 초기화
        episode_action_count = 0
        episode_masked_count = 0
        last_action = -1  # 정책 스위치 추적용

        unique_available_lots_in_episode = set()
        unique_processed_lots_in_episode = set()

        simTime = 0
        is_first_iteration = True
        # ✅ (신규) 평균 재공(WIP) 계산을 위한 변수 초기화
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

                # ✅ (신규) 주기적으로 WIP(재공) 수량 기록
                # (예: 시뮬레이션 시간으로 1시간(3600초)마다 기록)
                if simTime > last_wip_log_time + 3600:
                    # get_current_lot()은 느리므로, 이미 get_state()에서 로드된 lot_state 사용
                    current_wip = len(plantsim_manager.lot_state)
                    wip_log.append(current_wip)
                    last_wip_log_time = simTime


                if simTime > simulation_end_time:
                    break

                # ✅ (신규) Step 1: L1 에이전트를 통한 빠른 제약 조건 확인
                state_summary = plantsim_manager.get_state_summary()
                # Python 룰에 따라 '즉시' 금지된 행동을 가져옵니다.
                action_mask = safety_agent.get_action_mask(state_summary)


                # ###################################
                # ✅ Step 2: Action 결정 (RL 또는 안전 룰)
                # ###################################
                get_action_start_time = time.time()

                # 2-1. RL 에이전트의 기본 Action 선택
                action, epsilon = get_action(state, prev_epsilon, action_mask)
                ACTION_SOURCE = 'RL'  # (이제 RL이 항상 '안전한' 행동만 선택함)


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
                # 🚨 [추가] L1 마스킹 발생 시 카운트 및 정책 스위치 카운트
                current_action = action
                episode_action_count += 1

                if current_action != last_action:
                    L1_strategy_switch_count += 1
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
                    logger.info('iplt')
                    logger.info(plantsim_manager.get_iplt_state())
                    logger.info('output')
                    logger.info(plantsim_manager.get_output_state())
                    logger.info('target')
                    logger.info(plantsim_manager.get_target_state())

                ##################################
                #        Calculate Reward        #
                ##################################
                calculate_reward_start_time = time.time()

                # 🚨 수정된 calculate_reward 함수 호출 (safety_weight 전달)
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

                # ✅ [수정] 고유 ID를 Set에 누적
                unique_available_lots_in_episode.update(available_lot_ids)
                unique_processed_lots_in_episode.update(processed_lot_ids)

                output_df = plantsim_manager.get_output_state()
                target_df = plantsim_manager.get_target_state()

                if iteration_log:
                    # ✅ (신규) 비동기 XRL을 위한 데이터 로깅
                    xrl_data = {
                        "simTime": simTime,
                        "action": action,
                        "source": ACTION_SOURCE,  # (항상 'RL'이겠지만, 기록용)
                        "reward_components": reward_components
                    }
                    logger.info(f"[XRL_DATA] {json.dumps(xrl_data, cls=NumpyJSONEncoder)}")

                if not is_first_iteration:
                    dqn.buffer.add((state, action, reward, next_state))
                else:
                    is_first_iteration = False

                state = next_state

                if len(dqn.buffer.buffer) > batch_size:
                    loss = dqn.update(batch_size)
                    losses.append(loss)

                iteration_cnt = iteration_cnt + 1
                # 일정 주기마다 Target Network 업데이트
                if iteration_cnt % update_target_steps == 0:
                    dqn.target_net.load_state_dict(dqn.act_net.state_dict())

                update_network_end_time = time.time()
                update_network_elapsed_time = update_network_end_time - update_network_start_time
                total_update_network_elapsed_time += update_network_elapsed_time

                print_progress_bar(simTime, episode, num_episodes)

        summary_start_time = time.time()
        episode_rewards.append(total_reward)
        production_rewards.append(total_target)
        risk_reward.append(total_urgent_lot) # 👈 가중치 적용된 값 사용
        lot_processing_rewards.append(total_lot_move)

        priority_rewards.append(total_priority_lot)

        # chart_frequency 에피소드마다 그래프 업데이트
        if (episode + 1) % chart_frequency == 0 or (episode > num_episodes - 2):
            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(range(1, len(episode_rewards) + 1), episode_rewards, label="Total Reward")
            ax.plot(range(1, len(production_rewards) + 1), production_rewards, label="Target Reward")
            ax.plot(range(1, len(risk_reward) + 1), risk_reward, label="IPLT Reward")
            ax.plot(range(1, len(lot_processing_rewards) + 1), lot_processing_rewards, label="Lot Move")
            ax.plot(range(1, len(priority_rewards) + 1), priority_rewards, label="Priority Reward")

            ax.set_xlabel("Episodes")
            ax.set_ylabel("Reward")
            ax.set_title("Reward Components Over Episodes")
            ax.legend()

            graph_filename = os.path.join(plot_dir, f"reward_plot_{episode + 1}.png")
            plt.savefig(graph_filename)  # 파일로 저장!

            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(losses) + 1), losses, label='Training Loss', color='blue')
            plt.xlabel("Episodes")
            plt.ylabel("Loss")
            plt.title("Loss Curve Over Episodes")
            plt.legend()
            graph_filename = os.path.join(plot_dir, f"loss_plot_{episode + 1}.png")
            plt.savefig(graph_filename)

            eqp_history = plantsim_manager.get_value("var_eqp_history")

            chart_generator = ChartGenerator(plot_dir)
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

            # 총 생산량: comparison_df의 qty를 20으로 나눈 후 합산
            total_production = (target_df["qty"] / 20).sum()

            # IPLT 초과: repair_df의 count 합산
            iplt_exceed = repair_df["count"].sum()

            # IPLT 만족: repair_df의 device_id를 비교하여 해당하는 comparison_df의 qty를 20으로 나눈 후 합산
            satisfied_devices = target_df[target_df["device_id"].isin(repair_df["device_id"])]
            iplt_satisfied = (satisfied_devices["qty"] / 20).sum()

            # Target 달성: 모든 qty의 합을 모든 mat_qty의 합으로 나눈 후 100을 곱해 퍼센트로 환산
            target_achievement = (target_df["qty"].sum() / target_df["mat_qty"].sum()) * 100

            # ✅ (신규) 평균 재공(WIP) 계산
            avg_wip = np.mean(wip_log) if wip_log else 0
            # === U-Lot Cycle Time Calculation ===
            # 🚨 Lot ID가 'U'로 시작하는 Lot만 필터링
            ulots_df = lot_history_df[lot_history_df['lot_id'].str.startswith('U')].copy()

            if not ulots_df.empty:
                # Lot ID별 최소 (시작) 시간과 최대 (종료) 시간 계산
                # 'simTime'이 로트의 히스토리 시간을 나타낸다고 가정
                ulot_timing = ulots_df.groupby('lot_id')['sim_time'].agg(['min', 'max']).reset_index()

                # Cycle Time 계산 (Max Time - Min Time) 후 평균 계산
                avg_ulot_cycle_time = (ulot_timing['max'] - ulot_timing['min']).mean()
            else:
                avg_ulot_cycle_time = 0.0
            # === 🚨 [추가] Process 5 평균 버퍼 대기 시간 계산 ===
                # 🚨 [추가] Process 4와 Process 5의 평균 처리 시간 계산
            AVG_P4_PROC_TIME = calculate_avg_processing_time(lot_history_df, target_process_id=4)
            AVG_P5_PROC_TIME = calculate_avg_processing_time(lot_history_df, target_process_id=5)

            # 🚨 [추가] 처리 시간 변화량 KPI 계산
            if not np.isnan(AVG_P4_PROC_TIME) and not np.isnan(AVG_P5_PROC_TIME):
                T_DELTA_P5_P4 = AVG_P5_PROC_TIME - AVG_P4_PROC_TIME
            else:
                T_DELTA_P5_P4 = np.nan  # 데이터가 없을 경우 NaN 처리
            result = {
                "Total Outputs": float(total_production),
                "IPLT Over": int(iplt_exceed),
                "IPLT Meet": float(iplt_satisfied),
                "Target Meet": float(target_achievement)
            }

            logger.info(result)
            # ✅ [수정] '고유값' Set의 크기를 기준으로 최종 집계
            total_moved_priority_lots = len(unique_processed_lots_in_episode)
            total_available_priority_lots = len(unique_available_lots_in_episode)

            priority_rate = (total_moved_priority_lots / total_available_priority_lots) * 100 if total_available_priority_lots > 0 else 0

            # ✅ (신규) 해당 에피소드의 모든 결과를 하나의 딕셔너리로 종합
            episode_summary = {
                'episode': episode + 1,
                'total_reward': total_reward,
                'target_reward': total_target,
                'iplt_reward': total_urgent_lot,  # (기존 risk_reward)
                'lot_move_reward': total_lot_move,
                'priority_reward': total_priority_lot,
                'total_outputs': result.get('Total Outputs', 0),
                'iplt_over_count': result.get('IPLT Over', 0),
                'target_meet_rate': result.get('Target Meet', 0),
                'avg_wip': avg_wip,
                'priority_lots_processed': total_moved_priority_lots,  # ✅ [추가] 실제 처리 개수 (e.g. 50)
                'priority_lots_available': total_available_priority_lots,  # ✅ [추가] 처리 기회 총합 (e.g. 60)
                'priority_process_rate': priority_rate,
                'AVG_ULOT_CYCLE_TIME': avg_ulot_cycle_time,  # 🚨 [추가] ULot의 평균 총 소요 시간
                #'T_DELTA_P5_P4': T_DELTA_P5_P4,            # 🚨 처리 시간 변화량 KPI (P5 - P4)
                'L2_STRATEGY_COUNT': llm_analyst.L2_strategy_applied_count,  # LLM이 규칙을 보낸 총 횟수
                'POLICY_SWITCH_COUNT': L1_strategy_switch_count,  # 액션이 전환된 총 횟수
                'TOTAL_ACTION_STEPS': episode_action_count,  # 해당 에피소드의 총 액션 스텝
            }
            L1_strategy_switch_count = 0
            # ✅ (신규) 종합 결과를 메인 리스트에 추가
            all_episode_results.append(episode_summary)

        summary_end_time = time.time()
        summary_elapsed_time = summary_end_time - summary_start_time
        total_summary_elapsed_time += summary_elapsed_time

        episode_end_time = time.time()  #
        elapsed_time = episode_end_time - episode_start_time  # 소요 시간 계산

        if iteration_log:
            logger.info(
                'episode elapsed time : ' + str(elapsed_time) + ' total reward : ' + str(total_reward) + ' production : ' + str(total_target) + ' exceed : ' + str(total_urgent_lot) + ' lot_move : ' + str(total_lot_move))
            # ####################################################
            # ✅ (신규) L2 전략 계층 호출 (에피소드 종료 후 1회)
            # ####################################################
            logger.info(f"[L2] Episode {episode + 1} finished. Requesting strategic analysis...")
            # try:
            #     # L2 LLM에게 이번 에피소드 성과('result')와 현재 룰을 전달
            #     # safety_agent.current_rules는 List[SafetyRule] 입니다.
            #     # analyze_and_update_rules는 List[SafetyRule]를 반환합니다.
            #     # new_rules = llm_analyst.analyze_and_update_rules(
            #     #     episode_results=episode_summary,
            #     #     previous_rules=safety_agent.current_rules
            #     # )
            #     # logger.info(f"[L2] Received {len(new_rules)} new rules.")
            #     # for rule in new_rules:
            #     #     logger.info(f"[L2 RULE] {rule.model_dump_json(indent=2)}")
            #     # # ⬇️ (수정) '.get("rules")' 호출을 제거! ⬇️
            #     # # new_rules는 이미 List[SafetyRule]입니다.
            #     # safety_agent.set_rules(new_rules)
            #
            #     # ⬇️ (수정) 'safety_weight' 관련 로직 제거 (L1에서 이미 1.0으로 처리 중) ⬇️
            #     #logger.info(f"[L2] Strategy updated. New rules: {new_rules}")
            #
            # except Exception as e:
            #     # ⬇️ (수정) 오류 로깅 시 스택 트레이스 포함 ⬇️
            #     logger.error(f"[L2] Failed to update strategy.", exc_info=True)
            #     logger.info("Using previous settings.")  # (로그 추가)

    # df = pd.DataFrame({
    #     "episode_rewards": episode_rewards,
    #     "production_rewards": production_rewards,
    #     "risk_reward": risk_reward,
    #     "lot_processing_rewards": lot_processing_rewards
    # })
    #
    # # CSV 파일 저장
    # df.to_csv(folder_name + f"/csv/rewards_data.csv", index=False)
    # ✅ (신규) 모든 에피소드 결과를 담은 리스트로 데이터프레임 생성
    results_df = pd.DataFrame(all_episode_results)

    # CSV 파일 저장
    # ✅ (수정) df -> results_df
    results_df.to_csv(folder_name + f"/csv/rewards_data.csv", index=False)
    results_df.to_csv(f"{folder_name}/csv/results_{model_type}.csv", index=False)

    logger.info(f"get state : {total_get_state_elapsed_time:.4f}")
    logger.info(f"get action : {total_get_action_elapsed_time:.4f}")
    logger.info(f"execute action : {total_execute_action_elapsed_time:.4f}")
    logger.info(f"     lot_execute_time : {total_lot_execute_time:.4f}")
    logger.info(f"     eqp_execute_time : {total_eqp_execute_time:.4f}")
    logger.info(f"calculate reward : {total_calculate_reward_action_elapsed_time:.4f}")
    logger.info(f"update_network : {total_update_network_elapsed_time:.4f}")
    logger.info(f"summary : {total_summary_elapsed_time:.4f}")

    plantsim_manager.quit()
