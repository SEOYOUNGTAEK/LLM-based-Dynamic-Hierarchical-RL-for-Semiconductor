from collections import defaultdict
import pythoncom
import time
from plantsim.simulator.lot_manager import LotManager
from plantsim.simulator.eqp_manager import EqpManager
from plantsim.simulator.plan_manager import PlanManager
from plantsim.simulator.target_manager import TargetManager
from plantsim.simulator.iplt_manager import IPLTManager
import numpy as np
import json
import pandas as pd
from plantsim.plantsim import Plantsim
from logger import Logger
import orjson

class PlantsimManager:
    _instance = None  # ✅ Singleton 인스턴스 저장 변수
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def get_instance(cls):
        return cls._instance

    def resume_simulation(self):
        self.plantsim.execute_simtalk("resume_simulation")

    def init(self, iteration_log, version='23.2', visible=True, license_type='Educational'):
        pythoncom.CoInitialize()  # COM 객체 초기화
        self.logger = Logger().get_logger()
        self.plantsim = Plantsim(version=version, visible=visible, license_type=license_type)
        self.get_state_cnt = 0
        self.prev_target_state = None
        self.caculate_reward_cnt = 0
        self.iteration_log = iteration_log
        self.devices_achieved_goals = []

    def initialize_simulation(self, path, output_path, total_episodes):
        self.output_path = output_path
        self.eqp_manager = EqpManager()
        self.plantsim.load_model(path)
        self.total_episodes = total_episodes

        # 경로 설정 및 시뮬레이션 초기화
        self.plantsim.set_path_context(".Models.Model")  # 기본 경로 설정
        self.plantsim.set_event_controller()
        self.plantsim.reset_simulation()

        self.initialize_lot()
        # 시뮬레이션 종료 및 메모리 해제
        self.initialize_eqp(self.eqp_manager)
        self.initialize_input_plan()
        self.initialize_input_target()

        self.iplt_manager = IPLTManager()
        self.iplt_manager.read_targets_from_db()
        self.plantsim.start_simulation()

    def start_simulation(self, episode):
        self.plantsim.reset_simulation()
        self.plantsim.start_simulation()
        self.episode = episode

    def start_only_simulation(self):
        self.plantsim.start_simulation()

    def quit(self):
        self.plantsim.quit()

    def get_value(self, value):
        return self.plantsim.get_value(value)

    def execute_simtalk(self, value):
        return self.plantsim.execute_simtalk(value)

    def get_sim_time(self):
        return self.plantsim.get_value("EventController.simTime")

    def initialize_input_plan(self):
        self.plan_manager = PlanManager()
        self.plan_manager.read_plans_from_db()
        json_data = json.dumps(self.plan_manager.to_dict())
        self.plantsim.set_value('var_init_plan', str(json_data))
        self.plantsim.execute_simtalk("init_plan")
        self.plantsim.execute_simtalk("generate_plan")

    def initialize_input_target(self):
        self.target_manager = TargetManager()
        self.target_manager.read_targets_from_db()
        json_data = json.dumps(self.target_manager.to_dict())
        self.plantsim.set_value('var_init_target', str(json_data))
        self.plantsim.execute_simtalk("init_target")
        self.plantsim.execute_simtalk("generate_target")

    def get_state(self):
        current_lot_start = time.time()
        self.lot_state = self.get_current_lot()
        # print(f"current_lot_start 실행 시간: {time.time() - current_lot_start:.4f} 초")

        output_state = time.time()
        self.output_state = self.get_output_state()
        # print(f"output_state 실행 시간: {time.time() - output_state:.4f} 초")

        target_state = time.time()
        self.target_state = self.get_target_state(True)
        # print(f"target_state 실행 시간: {time.time() - target_state:.4f} 초")

        eqp_state = time.time()
        self.eqp_state = self.get_eqp_state()
        # print(f"eqp_state 실행 시간: {time.time() - eqp_state:.4f} 초")

        iplt_state = time.time()
        self.iplt_state = self.get_iplt_state()
        # print(f"iplt_state 실행 시간: {time.time() - iplt_state:.4f} 초")

        create_state = time.time()
        state, iplt_state_df = self.create_state_vector(self.lot_state, self.output_state, self.target_state, self.iplt_state)
        self.iplt_state = iplt_state_df
        # print(f"create_state 실행 시간: {time.time() - create_state:.4f} 초")
        return state

    def get_state_summary(self):
        """
        L1 Safety Agent가 룰을 적용하는 데 필요한
        '실시간' 메트릭을 계산하여 '문자열'로 반환합니다.
        (simple_safety_agent._parse_state_summary가 파싱할 수 있는 형태여야 함)
        """

        # 1. 'urgent_priority_lot_count' 계산
        # (get_state()가 이미 self.lot_state를 갱신했다고 가정)
        try:
            # 버퍼(process_id 5, IN_BUFFER)에 있는 Prio 1000 이상 Lot 개수
            priority_count = self.lot_state[
                (self.lot_state["is_urgent_lotid"]) & # ⬅️ [수정] Lot ID 기반 식별
                (self.lot_state["process_id"] == 5) &
                (self.lot_state["state"] == "IN_BUFFER")
                ].shape[0]
        except Exception:
            priority_count = 0  # self.lot_state가 비어있을 경우

        # 2. L1 에이전트가 파싱할 수 있는 형식의 문자열 생성
        # (참고: simple_safety_agent._parse_state_summary의 정규식과 일치해야 함)
        summary_str = f"urgent_priority_lot_count (Priority=1, ID starts with U): {priority_count}" # ⬅️ [수정] 출력 텍스트

        # (향후 'iplt_over_count' 같은 다른 메트릭도 여기에 추가할 수 있음)
        # summary_str += f"\n total_iplt_exceeded_count: {iplt_count}"

        return summary_str

    def normalization(self, data, max_val=None):
        if max_val is None:
            max_val = np.max(data)

        if max_val == 0:  # 최대값이 0이면 그대로 반환
            return np.zeros_like(data)

        return data / max_val

    def create_state_vector(self, lot_state_df, output_state_df, target_state_df, iplt_state_df):
        start_time = time.time()  # ⏱ 전체 실행 시간 측정 시작

        sim_time = self.get_sim_time()

        lot_state_count = (
            lot_state_df[(lot_state_df["process_id"] == 5) & (lot_state_df["state"] == "IN_BUFFER")]
            .groupby("device_id")
            .size()
            .reindex(index=["A", "B", "C", "D"], fill_value=0)
            .values
        )

        lot_state_vector = lot_state_count.flatten()

        ### ✅ Step 2: 목표 생산량 - 현재 생산량
        step2_start = time.time()
        output_dict = output_state_df.set_index("device_id")["qty"].to_dict()
        target_dict = target_state_df.set_index("device_id")["mat_qty"].to_dict()
        shortage_vector = np.array([target_dict.get(d, 0) - output_dict.get(d, 0) for d in ["A", "B", "C", "D"]])
        # print(f"Step 2 실행 시간: {time.time() - step2_start:.4f} 초")

        ### ✅ Step 3: IPLT 초과 위험 정보 계산
        step3_start = time.time()
        iplt_state_df = iplt_state_df.set_index("device_id")
        lot_state_df = lot_state_df.merge(iplt_state_df[["iplt_time"]], on="device_id", how="left").fillna(0)

        lot_state_df["current_iplt"] = np.where(
            lot_state_df["iplt_time"] == 0,  # IPLT 시간이 0이면
            0,  # current_iplt를 0으로 설정
            lot_state_df["sim_time"] - lot_state_df["dest_step_tkout_time"]  # 기존 로직 유지
        )

        # exceeded 계산 (IPLT가 0이면 exceeded=0 처리)
        lot_state_df["exceeded"] = np.where(
            lot_state_df["iplt_time"] == 0,  # IPLT 시간이 0이면
            0,  # exceeded는 0으로 설정
            (lot_state_df["current_iplt"] > lot_state_df["iplt_time"]).astype(int)  # 기존 로직 적용
        )

        def safe_mean(series):
            if series.empty or np.all(series.isna()):
                return 0  # 값이 없거나 NaN이면 0 반환
            return np.nanmean(series)

        exceeded_stats = lot_state_df.groupby("device_id").agg(
            remaining_lot_count=("exceeded", lambda x: len(x[lot_state_df["iplt_time"] > 0]) - x.sum() if np.any(
                lot_state_df["iplt_time"] > 0) else 0),
            remaining_avg=("current_iplt", lambda x: safe_mean(x[lot_state_df["exceeded"] == 0]) if np.any(
                lot_state_df["exceeded"] == 0) else 0),
        ).reindex(["A", "B", "C", "D"], fill_value=0)
        # print(f"Step 3 실행 시간: {time.time() - step3_start:.4f} 초")

        normalized_exceeded_stats = exceeded_stats.T  # 행과 열을 전환

        def normalize_row(row):
            max_val = row.max()
            min_val = row.min()
            return (row - min_val) / (max_val - min_val) if max_val > min_val else row

        normalized_exceeded_stats = normalized_exceeded_stats.apply(normalize_row, axis=1)

        ### ✅ Step 4: 정규화 및 병합
        step4_start = time.time()
        normalized_sim_time = self.normalization(np.array([sim_time]), max_val=86400)

        # --- ✅ (신규) State 4: '초 긴급 물량' 카운트 (4개) ---
        # Lot.priority 필드를 사용하여 긴급 물량(1000 이상)을 집계
        urgent_lot_count = (
            lot_state_df[
                (lot_state_df["process_id"] == 5) &
                (lot_state_df["state"] == "IN_BUFFER") &
                (lot_state_df["is_urgent_lotid"]) # ⬅️ [수정] Lot ID 기반 식별
                ]
            .groupby("device_id")
            .size()
            .reindex(index=["A", "B", "C", "D"], fill_value=0)
            .values
        )
        urgent_lot_vector = urgent_lot_count.flatten()  # (4개)

        lot_state_vector = self.normalization(lot_state_vector)
        shortage_vector = self.normalization(shortage_vector)
        iplt_vector = normalized_exceeded_stats.values.flatten()
        urgent_lot_vector = self.normalization(urgent_lot_vector) # (정규화)

        state_vector = np.concatenate([normalized_sim_time, lot_state_vector, shortage_vector, iplt_vector, urgent_lot_vector]).flatten()

        # print(f"Step 4 실행 시간: {time.time() - step4_start:.4f} 초")

        # print(f"⏱ 전체 실행 시간: {time.time() - start_time:.4f} 초")
        if self.iteration_log:
            self.logger.info(f'[DRL STATE] Urgent Lot Vector (Normalized): {state_vector[-4:]}')
        return state_vector, iplt_state_df.reset_index()

    def get_target_state(self, is_twin=False):
        target_state_df = pd.DataFrame(self.target_manager.to_dict())  # 기존 target_state 데이터프레임 생성

        # ✅ 목표를 달성한 device들의 mat_qty를 0으로 변경
        if is_twin:
            if hasattr(self, "devices_achieved_goals"):  # self.devices_achieved_goals이 존재하는지 확인
                target_state_df.loc[self.devices_achieved_goals, "mat_qty"] = 0

        return target_state_df

    def get_eqp_state(self):
        return pd.DataFrame(self.eqp_manager.to_dict())

    def get_iplt_state(self):
        return pd.DataFrame(self.iplt_manager.to_dict())

    def get_repair_qty(self):
        self.execute_simtalk("repair_to_json")
        repair = self.get_value("var_repair_qty")
        repair_dic = json.loads(repair)
        df = pd.DataFrame(repair_dic)
        count_df = df["device_id"].value_counts().reset_index()
        count_df.columns = ["device_id", "count"]
        return count_df

    def get_output_state(self):
        self.execute_simtalk("output_to_json")
        output = self.get_value("var_output")
        output_dict = json.loads(output)
        return pd.DataFrame(output_dict)

    def get_current_lot(self):
        self.execute_simtalk("current_lot_to_json")
        current_lot_str = self.get_value("var_current_lot")
        current_lot_dict = orjson.loads(current_lot_str)
        self.current_lot = pd.DataFrame.from_records(current_lot_dict)
        # 🚨 [필수 수정/추가] 원본 priority 값을 'priority_orig' 컬럼으로 저장
        # DRL 상태 계산 및 Action 9 로직에서 원본 긴급 Lot을 식별하기 위해 필요합니다.
        self.current_lot["is_urgent_lotid"] = self.current_lot["lot_id"].str.startswith('U')
        return self.current_lot

    def initialize_lot(self):
        self.lot_manager = LotManager()
        self.lot_manager.read_lots_from_db()
        json_data = json.dumps(self.lot_manager.to_dict())
        self.plantsim.set_value('var_init_lot', str(json_data))
        self.plantsim.execute_simtalk("init_lot")
        self.plantsim.execute_simtalk("generate_lot")

    def initialize_eqp(self, eqp_manager):
        self.eqp_manager.read_eqps_from_db()
        json_data = json.dumps(self.eqp_manager.to_dict())
        self.plantsim.set_value('var_init_eqp', str(json_data))
        self.plantsim.execute_simtalk("init_eqp")
        self.plantsim.execute_simtalk("generate_eqp")

    def get_total_equipment(self, process_id=None):
        return self.eqp_manager.get_total_equipment(process_id)

    def allocate_process_id(self, process_id, allocation):
        return self.eqp_manager.allocate_process_id(process_id, allocation)

    def set_eqp_allocation(self, allocation):
        json_data = json.dumps(allocation)
        self.plantsim.set_value('var_eqp_allocation', str(json_data))
        self.plantsim.execute_simtalk("eqp_device_change")

    def set_lot_priority(self, priority):
        json_data = json.dumps(priority)
        self.plantsim.set_value('var_lot_priority', str(json_data))
        self.plantsim.execute_simtalk("lot_priority_change")

    # def allocate_equipment(self, data, process_id, parameter, total_equipment, iplt_weight):
    #     # 필터링: ProcessID가 4 또는 5, 그리고 ProcessID가 5인 경우 Processing 상태 제외
    #     filtered_lot = self.current_lot[
    #         ((self.current_lot["process_id"] == 4) | (self.current_lot["process_id"] == 5)) &
    #         ~((self.current_lot["process_id"] == 5) & (self.current_lot["state"] == "PROCESSING"))
    #         ].copy()
    #
    #     # actual_iplt_time 계산
    #     filtered_lot["actual_iplt_time"] = filtered_lot["sim_time"] - filtered_lot["dest_step_tkout_time"]
    #
    #     iplt_time_dict = self.iplt_state.set_index("device_id")["iplt_time"].to_dict()
    #
    #     # apply를 사용하여 iplt_risk 계산
    #     filtered_lot["iplt_risk"] = filtered_lot.apply(
    #         lambda row: 0 if iplt_time_dict.get(row["device_id"], 0) == 0
    #         else row["actual_iplt_time"] / iplt_time_dict.get(row["device_id"], 0), axis=1
    #     )
    #
    #     # 각 device_id별로 iplt_risk > 0.7인 Lot 수를 계산하여 딕셔너리로 변환
    #     result_dict = (
    #         filtered_lot[filtered_lot["iplt_risk"] > 0.7]
    #         .groupby("device_id")
    #         .size()
    #         .to_dict()
    #     )
    #
    #     # ✅ process_id가 4인 경우: In Buffer + Processing 합산, 5인 경우: In Buffer만 합산
    #     filtered_data_4 = data[data["process_id"] == 4]
    #     filtered_data_5 = data[data["process_id"] == 5]
    #
    #     # ✅ device_id별로 Lot 개수 집계
    #     device_lot_counts = defaultdict(int)
    #
    #     for device, buffer_val, processing_val in zip(
    #             filtered_data_4["device_id"], filtered_data_4["In Buffer"], filtered_data_4["Processing"]
    #     ):
    #         device_lot_counts[device] += buffer_val + processing_val  # ✅ process_id 4는 Buffer + Processing 포함
    #
    #     for device, buffer_val in zip(filtered_data_5["device_id"], filtered_data_5["In Buffer"]):
    #         device_lot_counts[device] += buffer_val  # ✅ process_id 5는 Buffer만 포함
    #
    #     org_lot_counts = device_lot_counts.copy()
    #     # ✅ iplt_risk 개수도 Lot 개수에 합산 (iplt_weight 적용)
    #     for device, risk_count in result_dict.items():
    #         device_lot_counts[device] += risk_count * iplt_weight  # ✅ iplt_risk 반영
    #
    #     if self.iteration_log:
    #         self.logger.info('org_lot_counts')
    #         self.logger.info(org_lot_counts)
    #         self.logger.info('device_lot_counts')
    #         self.logger.info(device_lot_counts)
    #
    #     # ✅ 전체 Lot 수 계산
    #     total_lots = sum(device_lot_counts.values())
    #
    #     # ✅ 초기 할당: 비율 기반으로 장비 개수 계산
    #     allocated_equipment = {
    #         device: max(1, int(round((lot_count / total_lots) * total_equipment)))  # 최소 1개 보장
    #         for device, lot_count in device_lot_counts.items()
    #     }
    #
    #     # ✅ 총 합 체크 및 조정
    #     allocated_sum = sum(allocated_equipment.values())
    #
    #     # ✅ 초과된 경우 → 가장 많은 장비를 받은 디바이스부터 하나씩 줄임
    #     if allocated_sum > total_equipment:
    #         excess = allocated_sum - total_equipment
    #         sorted_devices = sorted(allocated_equipment, key=lambda x: -allocated_equipment[x])  # 많이 받은 순 정렬
    #         for device in sorted_devices:
    #             if excess <= 0:
    #                 break
    #             if allocated_equipment[device] > 1:  # 최소 1개 보장
    #                 allocated_equipment[device] -= 1
    #                 excess -= 1
    #
    #     # ✅ 부족한 경우 → 가장 적게 받은 디바이스부터 하나씩 추가
    #     elif allocated_sum < total_equipment:
    #         deficit = total_equipment - allocated_sum
    #         sorted_devices = sorted(allocated_equipment, key=lambda x: allocated_equipment[x])  # 적게 받은 순 정렬
    #         for device in sorted_devices:
    #             if deficit <= 0:
    #                 break
    #             allocated_equipment[device] += 1
    #             deficit -= 1
    #
    #     return allocated_equipment
    def allocate_equipment(self, data, process_id, parameter, total_equipment, urgent_weight, iplt_weight):

        # --- 1. 'IPLT 위험 Lot' 계산 (iplt_weight > 0 일 때만 필요) ---
        iplt_risk_dict = {}
        if iplt_weight > 0:
            filtered_lot = self.current_lot[
                ((self.current_lot["process_id"] == 4) | (self.current_lot["process_id"] == 5)) &
                ~((self.current_lot["process_id"] == 5) & (self.current_lot["state"] == "PROCESSING"))
                ].copy()

            if not filtered_lot.empty:
                filtered_lot["actual_iplt_time"] = filtered_lot["sim_time"] - filtered_lot["dest_step_tkout_time"]
                iplt_time_dict = self.iplt_state.set_index("device_id")["iplt_time"].to_dict()

                filtered_lot["iplt_risk"] = filtered_lot.apply(
                    lambda row: 0 if iplt_time_dict.get(row["device_id"], 0) == 0
                    else row["actual_iplt_time"] / iplt_time_dict.get(row["device_id"], 0), axis=1
                )

                # IPLT 위험 기준치 (e.g., 70% 초과)
                iplt_risk_dict = (
                    filtered_lot[filtered_lot["iplt_risk"] > 0.7]
                    .groupby("device_id")
                    .size()
                    .to_dict()
                )

        # --- 2. '초 긴급 Lot' 계산 (urgent_weight > 0 일 때만 필요) ---
        urgent_lot_dict = {}
        if urgent_weight > 0:
            try:
                urgent_lots_in_buffer = self.current_lot[
                    (self.current_lot["process_id"] == 5) &
                    (self.current_lot["state"] == "IN_BUFFER") &
                    (self.current_lot["priority"] >= 1000)
                    ]
                urgent_lot_dict = (
                    urgent_lots_in_buffer
                    .groupby("device_id")
                    .size()
                    .to_dict()
                )
            except KeyError:
                urgent_lot_dict = {}

        # --- 3. '총 WIP' 계산 (기본) ---
        filtered_data_4 = data[data["process_id"] == 4]
        filtered_data_5 = data[data["process_id"] == 5]
        device_lot_counts = defaultdict(int)

        buffer_col = "In Buffer" if "In Buffer" in filtered_data_4.columns else None
        processing_col = "Processing" if "Processing" in filtered_data_4.columns else None
        buffer_col_5 = "In Buffer" if "In Buffer" in filtered_data_5.columns else None

        if buffer_col and processing_col:
            for device, buffer_val, processing_val in zip(
                    filtered_data_4["device_id"], filtered_data_4[buffer_col], filtered_data_4[processing_col]
            ):
                device_lot_counts[device] += buffer_val + processing_val

        if buffer_col_5:
            for device, buffer_val in zip(filtered_data_5["device_id"], filtered_data_5[buffer_col_5]):
                device_lot_counts[device] += buffer_val

        org_lot_counts = device_lot_counts.copy()

        # --- 4. ✅ [수정] '긴급 Lot' 가중치와 'IPLT 위험 Lot' 가중치를 모두 합산 ---
        # 모든 Device 목록 (WIP, 긴급, IPLT 중 하나라도 있는 경우)
        devices = set(list(device_lot_counts.keys()) + list(urgent_lot_dict.keys()) + list(iplt_risk_dict.keys()))
        target_devices = ["A", "B", "C", "D"]  # 할당 대상이 되는 device 목록

        for device in target_devices:
            if device not in device_lot_counts:  # WIP이 0인 Device도 초기화
                device_lot_counts[device] = 0

            urgent_bonus = urgent_lot_dict.get(device, 0) * urgent_weight
            iplt_bonus = iplt_risk_dict.get(device, 0) * iplt_weight
            device_lot_counts[device] += urgent_bonus + iplt_bonus

        # device_lot_counts에 target_devices 외의 것이 있다면 제거
        device_lot_counts = defaultdict(int, {k: v for k, v in device_lot_counts.items() if k in target_devices})

        if self.iteration_log:
            self.logger.info('org_lot_counts (WIP only)')
            self.logger.info(org_lot_counts)
            self.logger.info(f'urgent_lot_counts (Weight: {urgent_weight}): {urgent_lot_dict}')
            self.logger.info(f'iplt_risk_counts (Weight: {iplt_weight}): {iplt_risk_dict}')
            self.logger.info(f'weighted_device_lot_counts (WIP + Bonuses)')
            self.logger.info(device_lot_counts)

        # --- 5. ❗️[수정된] 설비 할당 로직 ---
        target_devices = ["A", "B", "C", "D"]
        num_devices = len(target_devices)

        # 5-1. (신규) 모든 디바이스에 최소 1개씩 기본 할당
        if total_equipment < num_devices:
            # (설비가 4대 미만인 극단적 경우)
            allocated_equipment = {device: 0 for device in target_devices}
            for i in range(total_equipment):
                allocated_equipment[target_devices[i]] = 1
            return allocated_equipment

        allocated_equipment = {device: 1 for device in target_devices}
        remaining_equipment = total_equipment - num_devices  # (e.g., 12 - 4 = 8)

        # 5-2. 가중치 합산 (WIP + 보너스)
        weighted_scores = defaultdict(float)
        total_score = 0
        for device in target_devices:
            score = device_lot_counts.get(device, 0)
            weighted_scores[device] = score
            total_score += score

        if total_score == 0:
            # 5-3. (신규) WIP이 0이면 남은 설비를 균등 배분
            base_alloc = remaining_equipment // num_devices
            remainder = remaining_equipment % num_devices
            for i in range(num_devices):
                allocated_equipment[target_devices[i]] += base_alloc
                if i < remainder:
                    allocated_equipment[target_devices[i]] += 1
            return allocated_equipment

        # 5-4. 가중치 비율에 따라 남은 설비(8대) 배분
        for device in target_devices:
            if total_score > 0:
                share = weighted_scores[device] / total_score
                bonus_alloc = int(round(share * remaining_equipment))
                allocated_equipment[device] += bonus_alloc

        # 5-5. (신규) 할당 총합이 total_equipment와 일치하도록 보정
        current_sum = sum(allocated_equipment.values())

        if current_sum < total_equipment:
            deficit = total_equipment - current_sum
            # 점수가 높은 순으로 1대씩 추가
            sorted_devices = sorted(weighted_scores, key=weighted_scores.get, reverse=True)
            for i in range(deficit):
                allocated_equipment[sorted_devices[i % num_devices]] += 1

        elif current_sum > total_equipment:
            excess = current_sum - total_equipment
            # (점수가 낮은 순으로) 1대씩 제거 (단, 1개는 보장)
            sorted_devices = sorted(weighted_scores, key=weighted_scores.get)
            for i in range(excess):
                device_to_remove = sorted_devices[i % num_devices]
                if allocated_equipment[device_to_remove] > 1:  # 1대는 보장
                    allocated_equipment[device_to_remove] -= 1
                else:
                    # 1대밖에 없으면, 점수 낮은 다음 디바이스에서 제거
                    next_device = sorted_devices[(i + 1) % num_devices]
                    if allocated_equipment[next_device] > 1:
                        allocated_equipment[next_device] -= 1
                    else:
                        # (극단적 상황) 1대씩만 할당된 경우, 점수 가장 낮은 곳에서 제거
                        allocated_equipment[sorted_devices[0]] -= 1

                        # 할당 딕셔너리에 4개 device가 모두 포함되도록 보장 (최종 확인)
        for device in target_devices:
            if device not in allocated_equipment:
                allocated_equipment[device] = 0  # (이론상 실행되지 않아야 함)

        if self.iteration_log:
            self.logger.info(f'Final Allocation (Min 1 Guaranteed): {allocated_equipment}')

        return allocated_equipment
    def execute_action(self, action, simTime):
        self.save_prev_state()

        if self.iteration_log:
            self.logger.info("Before lot_priority_process SimTime : " + str(self.get_sim_time()))

        lot_start_time = time.time()
        self.lot_priority_process(action, simTime)
        lot_end_time = time.time()
        lot_elapsed_time = lot_end_time - lot_start_time

        if self.iteration_log:
            self.logger.info("Before eqp_allocation_process SimTime : " + str(self.get_sim_time()))

        eqp_start_time = time.time()
        self.eqp_allocation_process(action)
        eqp_end_time = time.time()
        eqp_elapsed_time = eqp_end_time - eqp_start_time

        if self.iteration_log:
            self.logger.info("Before resume_simulation SimTime : " + str(self.get_sim_time()))

        self.plantsim.execute_simtalk("reset_process_e")
        self.plantsim.execute_simtalk("resume_simulation")
        return lot_elapsed_time, eqp_elapsed_time

    def calculate_priority(self, row, target_state, output_state, iplt_state, simTime, target_weight=1.0, ):
        iplt_weight = 1.0 - target_weight
        device = row["device_id"]

        # ✅ target_state와 output_state 필터링 최적화
        target_data = target_state.loc[target_state["device_id"] == device]
        output_data = output_state.loc[output_state["device_id"] == device]

        # ✅ target_deficit 계산 최적화
        if not target_data.empty and not output_data.empty:
            target_qty = target_data.iloc[0]["mat_qty"]
            remain_qty = target_qty - output_data.iloc[0]["qty"]
            target_deficit = remain_qty / target_qty if target_qty > 0 else remain_qty
        else:
            target_deficit = 0  # 기본값

        target_score = target_deficit * 5000

        # ✅ IPLT 점수 계산 최적화
        iplt_data = iplt_state.loc[iplt_state["device_id"] == device]

        if not iplt_data.empty:
            iplt_time = iplt_data.iloc[0]["iplt_time"]
            current_iplt = simTime - row["dest_step_tkout_time"]
            iplt_score = current_iplt / iplt_time if iplt_time > 0 else float("inf")
        else:
            iplt_time = 0
            iplt_score = float("inf")

        # ✅ 최종 점수 계산
        total_weighted_score = target_weight * target_score + iplt_weight * iplt_score

        return target_score, target_weight, iplt_score, iplt_weight, iplt_time, total_weighted_score, simTime

    def lot_priority_process(self, action, simTime):
        # ✅ Lot 필터링 최적화
        global df_sorted
        filtered_lot = self.current_lot.loc[self.current_lot["process_id"] <= 5].copy()

        if filtered_lot.empty:
            return {}, pd.DataFrame()  # 빈 DataFrame 반환

        # 🚨 [추가] 원본 Priority 값을 'priority_orig' 컬럼으로 확보하는 로직 🚨
        # 이 값은 Lot 객체가 DB에서 로드될 때 가지고 있던 Priority 값 (1 또는 5)입니다.
        if "priority_orig" not in filtered_lot.columns:
            filtered_lot["priority_orig"] = filtered_lot["priority"]

        # ✅ target_qty 가져오기 (merge 대신 map 사용)
        target_qty_map = self.target_state.set_index('device_id')["mat_qty"].to_dict()
        filtered_lot["target_qty"] = filtered_lot["device_id"].map(target_qty_map).fillna(0)

        # ✅ remain_qty 계산 (merge 제거)
        output_qty_map = self.output_state.set_index('device_id')['qty'].to_dict()
        filtered_lot["remain_qty"] = filtered_lot["target_qty"] - filtered_lot["device_id"].map(output_qty_map).fillna(
            0)

        # ✅ actual_iplt_time 계산
        filtered_lot["actual_iplt_time"] = simTime - filtered_lot["dest_step_tkout_time"]

        # ✅ IPLT 초과 여부 계산 (merge 없이 map 사용)
        iplt_time_map = self.iplt_state.set_index("device_id")["iplt_time"].to_dict()
        filtered_lot["iplt_time"] = filtered_lot["device_id"].map(iplt_time_map).fillna(0)  # NaN이면 0으로 변경
        filtered_lot["remain_iplt_time"] = filtered_lot["iplt_time"] - filtered_lot["actual_iplt_time"]
        filtered_lot.loc[filtered_lot["iplt_time"] == 0, "remain_iplt_time"] = 99999

        # ✅ IPLT 초과 여부 계산 (0이면 무조건 False)
        filtered_lot["is_iplt_over"] = filtered_lot["actual_iplt_time"].gt(filtered_lot["iplt_time"]) & filtered_lot[
            "iplt_time"].gt(0)

        # ✅ `iplt_time == 0`이면 안전한 값으로 변경 (ZeroDivision 방지)
        iplt_score = np.divide(
            filtered_lot["actual_iplt_time"],
            filtered_lot["iplt_time"],
            out=np.zeros_like(filtered_lot["actual_iplt_time"]),  # 기본값 0으로 설정
            where=filtered_lot["iplt_time"] > 0  # 0인 경우 연산 제외
        )

        iplt_score = np.nan_to_num(iplt_score, nan=0.0)

        if action in [0, 1, 2]:
            df_sorted = filtered_lot.sort_values(
                by=["actual_iplt_time"],
                ascending=[False]
            ).reset_index(drop=True)
            # ❗️[통합] IPLT 초과 Lot을 -1로 설정
            df_sorted["priority"] = np.where(df_sorted["is_iplt_over"], -1, np.arange(1, len(df_sorted) + 1))

        elif action in [3, 4, 5]:
            df_sorted = filtered_lot.sort_values(
                by=["remain_qty"],
                ascending=[False]
            ).reset_index(drop=True)
            # ❗️[통합] IPLT 초과 Lot을 -1로 설정
            df_sorted["priority"] = np.where(df_sorted["is_iplt_over"], -1, np.arange(1, len(df_sorted) + 1))

        elif action in [6, 7, 8]:
            df_sorted = filtered_lot.sort_values(
                by=["remain_iplt_time"],
                ascending=[True]
            ).reset_index(drop=True)
            # ❗️[통합] IPLT 초과 Lot을 -1로 설정
            df_sorted["priority"] = np.where(df_sorted["is_iplt_over"], -1, np.arange(1, len(df_sorted) + 1))

        elif action == 9:

            df_sorted = filtered_lot.sort_values(

                by=["remain_iplt_time"],

                ascending=[True]

            ).reset_index(drop=True)

            # 1. 기본 순서: Action 9 정렬 결과대로 1, 2, 3... 부여
            priority_base = np.arange(1, len(df_sorted) + 1)

            # 2. IPLT 초과: -1 부여
            priority_iplt = np.where(df_sorted["is_iplt_over"], -1, priority_base)
            df_sorted["priority"] = np.where(df_sorted["is_urgent_lotid"], -10,priority_iplt)

        # --- 모든 Action이 이 공통 로직을 사용하게 됩니다 ---

        # ✅ Dictionary 생성 최적화
        lot_priority = df_sorted.set_index("lot_id")["priority"].to_dict()

        # (디버깅 로그가 필요하면 여기에 추가)
        # if self.iteration_log:
        #      self.logger.info("lot_priority")
        #      self.logger.info(str(lot_priority))
        #      df_sorted_temp = df_sorted[
        #          ["lot_id", "device_id", "actual_iplt_time", "is_iplt_over", "priority","remain_iplt_time"]]
        #      self.logger.info(df_sorted_temp.to_string(index=False))

        self.set_lot_priority(lot_priority)
        return lot_priority, df_sorted

    # def eqp_allocation_process(self, action):
    #     if action in [0, 3, 6] or action == 9:
    #         parameter = 1
    #     elif action in [1, 4, 7]:
    #         parameter = 2
    #     elif action in [2, 5, 8]:
    #         parameter = 3
    #
    #     summary_start_time = time.time()
    #     summary = (
    #         self.current_lot
    #         .groupby(['process_id', 'device_id', 'state'])
    #         .size()
    #         .unstack(fill_value=0)
    #         .reset_index()
    #     )
    #     summary.columns.name = None
    #     summary.rename(columns={'IN_BUFFER': 'In Buffer', 'PROCESSING': 'Processing'}, inplace=True)
    #     data = pd.DataFrame(summary)
    #     summary_end_time = time.time()
    #     summary_elapsed_time = summary_end_time - summary_start_time
    #
    #     # 설비 할당 계산
    #     allocate_start_time = time.time()
    #     total_equipment = self.get_total_equipment(process_id=5)
    #     allocation = self.allocate_equipment(data, process_id=5, parameter=parameter,
    #                                          total_equipment=total_equipment, iplt_weight=0)
    #     allocate_end_time = time.time()
    #     allocate_elapsed_time = allocate_end_time - allocate_start_time
    #
    #     # 장비 할당 결과 처리
    #     process_start_time = time.time()
    #     allocate_result = self.allocate_process_id(process_id=5, allocation=allocation)
    #     process_end_time = time.time()
    #     process_elapsed_time = process_end_time - process_start_time
    #
    #     if self.iteration_log:
    #         self.logger.info('allocate_result')
    #         self.logger.info(str(allocate_result))
    #
    #     self.set_eqp_allocation(allocate_result)
    def eqp_allocation_process(self, action):
        if action in [0, 3, 6] or action == 9:
            parameter = 1
        elif action in [1, 4, 7]:
            parameter = 2
        elif action in [2, 5, 8]:
            parameter = 3

        summary_start_time = time.time()
        summary = (
            self.current_lot
            .groupby(['process_id', 'device_id', 'state'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        summary.columns.name = None
        summary.rename(columns={'IN_BUFFER': 'In Buffer', 'PROCESSING': 'Processing'}, inplace=True)
        data = pd.DataFrame(summary)
        summary_end_time = time.time()
        summary_elapsed_time = summary_end_time - summary_start_time

        # --- ✅ [신규] Action에 따른 동적 가중치 설정 ---
        current_urgent_weight = 0.0
        current_iplt_weight = 0.0  # ⬅️ IPLT 가중치 초기화

        if action == 9:
            # '초 긴급 물량' 처리 시, '긴급 Lot'에 가중치
            current_urgent_weight = 50.0  # (가중치 값은 조절 가능)
        elif action in [6, 7, 8]:
            # 'IPLT' 관리 시, 'IPLT 위험 Lot'에 가중치
            current_iplt_weight = 50.0  # (가중치 값은 조절 가능)
        # -----------------------------------------------

        # 설비 할당 계산
        allocate_start_time = time.time()
        total_equipment = self.get_total_equipment(process_id=5)

        # ✅ [수정] allocate_equipment 함수에 두 가중치를 모두 전달
        allocation = self.allocate_equipment(data, process_id=5, parameter=parameter,
                                             total_equipment=total_equipment,
                                             urgent_weight=current_urgent_weight,
                                             iplt_weight=current_iplt_weight)  # ⬅️ 수정

        allocate_end_time = time.time()
        allocate_elapsed_time = allocate_end_time - allocate_start_time

        # 장비 할당 결과 처리
        process_start_time = time.time()
        allocate_result = self.allocate_process_id(process_id=5, allocation=allocation)
        process_end_time = time.time()
        process_elapsed_time = process_end_time - process_start_time

        if self.iteration_log:
            self.logger.info('allocate_result')
            self.logger.info(str(allocate_result))

        self.set_eqp_allocation(allocate_result)
    def save_prev_state(self):
        self.prev_lot_state = self.lot_state
        self.prev_output_state = self.output_state
        self.prev_target_state = self.target_state
        self.prev_eqp_state = self.eqp_state
        self.prev_iplt_state = self.iplt_state


    def calculate_reward(self, iteration_log):  # 👈 safety_weight 인수를 추가합니다.
        self.caculate_reward_cnt = self.caculate_reward_cnt + 1

        if self.prev_target_state is None:
            return 0, 0, 0, 0, 0, {}, 0, [], []

        target_reward = self.calculate_target_reward()
        lot_move_reward, lot_move_cnt = self.compute_lot_processing_reward()
        urgent_lot_reward, moved_risk_count, high_risk_lot_cnt = self.calculate_urgent_lot_reward()

        # ✅ (신규) '초 긴급 물량' 보상 계산
        priority_lot_reward, moved_priority_count, available_priority_count, \
            available_lot_ids, processed_lot_ids = self.calculate_priority_lot_reward()


        # ✅ (수정) 최종 Reward에 'priority_lot_reward' 추가
        reward = target_reward + lot_move_reward + urgent_lot_reward + priority_lot_reward

        reward_components = {
            "target_reward": target_reward,
            "lot_move_reward": lot_move_reward,
            "urgent_lot_reward": urgent_lot_reward,  # 가중치 적용 전 원본
            "priority_lot_reward": priority_lot_reward  # ⬅️ (신규)
        }

        if iteration_log:
            self.logger.info('reward : ' + str(reward))
            self.logger.info(f'lot_move_reward : {lot_move_reward:.4f} lot_move_cnt : {lot_move_cnt}')
            self.logger.info(f'target_reward : {target_reward:.4f}')

            self.logger.info(f'moved_risk_count : {moved_risk_count} high_risk_lot_cnt : {high_risk_lot_cnt}')
            self.logger.info(
                f'priority_lot_reward : {priority_lot_reward:.4f} (Processed: {moved_priority_count} / Available: {available_priority_count})')
            self.logger.info(f'priority_lot_reward (Net): {priority_lot_reward:.4f} (Processed: {moved_priority_count} / Available: {available_priority_count})')


        return reward, target_reward, lot_move_reward, urgent_lot_reward, priority_lot_reward, \
            reward_components, moved_priority_count, available_lot_ids, processed_lot_ids

        # ✅ (신규) calculate_priority_lot_reward 함수

    def calculate_priority_lot_reward(self):

        # 1. 이전 Lot 목록 (id: process_id)
        prev_lot_dict = self.prev_lot_state.set_index("lot_id")["process_id"].to_dict()


        urgent_lots_to_check = self.prev_lot_state[
            (self.prev_lot_state["process_id"] == 5) &
            (self.prev_lot_state["lot_id"].str.startswith('U')) # ⬅️ [수정] Lot ID 기반 식별
            ]
        available_count = len(urgent_lots_to_check)
        available_lot_ids = urgent_lots_to_check["lot_id"].tolist()

        # --- ❗️[디버깅 로그 2] ---
        if self.iteration_log:
            self.logger.info(f"[REWARD_DEBUG] 1. 처리 대상 '기회' Lot (Available): {available_count}개")


        # 3. 처리 '기회'가 없었으면 0점 반환
        if available_count == 0:
            if self.iteration_log:
                self.logger.info("[REWARD_DEBUG] === 기회 없음. 0점 반환. ===")
            return 0.0, 0, 0, [], []

        # --- ❗️[디버깅 로그 2B] ---
        if self.iteration_log:
            self.logger.info("[REWARD_DEBUG] 0B. '현재(Current)' Lot 상태 (긴급 Lot ID만 필터링):")
            try:
                current_lot_status = self.lot_state[
                    self.lot_state["lot_id"].isin(available_lot_ids)
                ][["lot_id", "process_id", "priority"]].reset_index(drop=True)

                if current_lot_status.empty:
                    self.logger.info("[REWARD_DEBUG]    -> '현재' 상태에 해당 Lot ID가 없음 (Process 5를 통과하여 소멸됨).")
                else:
                    self.logger.info("\n" + current_lot_status.to_string())
            except Exception as e:
                self.logger.error(f"[REWARD_DEBUG]    -> lot_state 로깅 실패: {e}")

        # 4. ❗️[로직 수정] "미처리된" Lot을 계산합니다.
        # (처리 기회가 있었는데[available_lot_ids], '현재' 상태에도 여전히 'process_id == 5'인 Lot)
        unprocessed_lots = self.lot_state[
            (self.lot_state["lot_id"].isin(available_lot_ids)) &
            (self.lot_state["process_id"] == 5)
            ]
        unprocessed_count = len(unprocessed_lots)

        # 5. ❗️[로직 수정] "처리된" Lot = 기회(Available) - 미처리(Unprocessed)
        processed_count = available_count - unprocessed_count

        # [로그용] 처리된 Lot ID 목록을 추적합니다.
        processed_lot_ids = []
        if processed_count > 0:
            unprocessed_ids_set = set(unprocessed_lots["lot_id"].tolist())
            processed_lot_ids = [lot_id for lot_id in available_lot_ids if lot_id not in unprocessed_ids_set]

        # --- ❗️[디버깅 로그 3] ---
        if self.iteration_log:
            self.logger.info(
                f"[REWARD_DEBUG] 2. '실제 처리'된 Lot (Processed): {processed_count}개  (계산: {available_count} - {unprocessed_count})")
            if processed_count > 0:
                self.logger.info(f"[REWARD_DEBUG]    Processed Lot IDs: {processed_lot_ids}")
            elif available_count > 0:
                self.logger.warning(f"[REWARD_DEBUG] ❗️경고: 처리 기회({available_count}개)가 있었으나, 처리된 Lot이 0개입니다.")

        # 6. [보상] 처리 성공 시 1개당 +200점
        priority_reward = processed_count * 50.0

        # 7. '지연 페널티' 계산
        priority_penalty = max(unprocessed_count * -10.0, -200.0)  # 페널티 -10점, 최대 -200으로 유지

        # --- ❗️[디버깅 로그 4] ---
        if self.iteration_log:
            self.logger.info(f"[REWARD_DEBUG] 3. 리워드 계산:")
            self.logger.info(f"[REWARD_DEBUG]    보상 (Success): {processed_count} * 200.0 = {priority_reward}")
            self.logger.info(
                f"[REWARD_DEBUG]    페널티 (Delay): {unprocessed_count} * -10.0 = {priority_penalty} (Max -200)")

        # 8. 최종 리워드 = 보상 + 페널티
        total_priority_reward = priority_reward + priority_penalty

        # --- ❗️[디버깅 로그 5] ---
        if self.iteration_log:
            self.logger.info(f"[REWARD_DEBUG] === 최종 Priority Reward: {total_priority_reward} ===")

        # 9. 고유 ID 반환
        return total_priority_reward, processed_count, available_count, available_lot_ids, processed_lot_ids
    def calculate_target_reward(self):
        # ✅ 1. 목표 생산량 달성 보상 (device별로 계산)
        prev_shortage = self.prev_target_state["mat_qty"] - self.prev_output_state["qty"]
        current_shortage = self.target_state["mat_qty"] - self.output_state["qty"]
        # ✅ 부족량 감소한 만큼 보상
        #target_reward = np.sum(prev_shortage - current_shortage) / 2000.0  # 보상 크기 조정 #일단 바꿈.. 이상해 이거 ㅋㅋㅋ + 타겟보너스도 이상하다 ㅎㅎ
        target_reward = np.sum(
            prev_shortage - current_shortage) / 200.0 # 일시적으로 바꿈 SYT ㅋㅋ LLM 보려고
        # ✅ 2. 목표 생산량 100% 달성 시 추가 보상
        #goal_bonus = 3  # 기본값
        # ✅ 목표를 달성한 device 찾기
        devices_achieved_goal = self.target_state[current_shortage <= 0].index.tolist()  # 목표 달성한 device 리스트
        if len(devices_achieved_goal) > 0:  # 하나라도 목표를 달성한 경우
            #goal_bonus = 10  # 보상 지급
            self.devices_achieved_goals.extend(devices_achieved_goal)
        # ✅ 최종 보상 계산
        target_reward = target_reward# + goal_bonus
        return target_reward

    def compute_lot_processing_reward(self):
        self.plantsim.execute_simtalk("process_e_to_json")
        process_increment = self.plantsim.get_value("var_process_e")

        lot_processing_reward = process_increment / 5

        return lot_processing_reward, process_increment

    def calculate_urgent_lot_reward(self):
        prev_lot_dict = self.prev_lot_state.set_index("lot_id")["process_id"].to_dict()
        iplt_dict = self.iplt_state.set_index("device_id")["iplt_time"].to_dict()

        # ✅ Process 5인 Lot만 선택
        process_5_lots = self.lot_state[self.lot_state["process_id"] == 5].copy()

        # ✅ IPLT 초과 위험 계산 (Process 5인 Lot 기준)
        process_5_lots["iplt_time"] = process_5_lots["device_id"].map(iplt_dict)
        process_5_lots["actual_iplt_time"] = process_5_lots["sim_time"] - process_5_lots["dest_step_tkout_time"]
        process_5_lots["iplt_risk"] = process_5_lots["actual_iplt_time"] / process_5_lots["iplt_time"]

        high_risk_lots = process_5_lots[(process_5_lots["iplt_risk"] >= 0.5) & (process_5_lots["iplt_risk"] < 1)]

        # ✅ 이동한 Lot 필터링 (Process ID가 증가한 Lot만 포함)
        moved_lots = process_5_lots[
            (process_5_lots["lot_id"].isin(prev_lot_dict.keys())) &
            (process_5_lots["process_id"] > process_5_lots["lot_id"].map(prev_lot_dict))
            ].copy()

        # ✅ 이동한 Lot 중에서 IPLT 초과 위험이 높은 Lot 개수 체크
        moved_urgent_lots = moved_lots[moved_lots["lot_id"].isin(high_risk_lots["lot_id"])]
        moved_count = len(moved_urgent_lots)

        if len(high_risk_lots) == 0:
            return 0, 0, 0

        # ✅ 보상 계산 (이동한 위험 Lot의 비율 기반)
        urgent_lot_reward = min((moved_count / len(high_risk_lots)) * 3.0, 2.0)
        return urgent_lot_reward, moved_count, len(high_risk_lots)

    def calculate_repair_lot_penalty(self):
        prev_lot_dict = self.prev_lot_state.set_index("lot_id")["process_id"].to_dict()
        iplt_dict = self.iplt_state.set_index("device_id")["iplt_time"].to_dict()

        # ✅ 전체 Lot에서 IPLT 초과 위험 계산 (이동한 Lot이 아니라 전체 Lot 기준으로!)
        self.lot_state["iplt_time"] = self.lot_state["device_id"].map(iplt_dict)
        self.lot_state["actual_iplt_time"] = self.lot_state["sim_time"] - self.lot_state["dest_step_tkout_time"]
        self.lot_state["iplt_risk"] = self.lot_state["actual_iplt_time"] / self.lot_state["iplt_time"]

        need_repair_lots = self.lot_state[self.lot_state["iplt_risk"] > 1]
        # ✅ IPLT 초과 위험이 높은 Lot 필터링 (전체 Lot 기준으로 상위 25%)

        # ✅ 이동한 Lot 필터링 (Process ID가 증가한 Lot만 포함)
        moved_lots = self.lot_state[
            (self.lot_state["lot_id"].isin(prev_lot_dict.keys())) &
            (self.lot_state["process_id"] > self.lot_state["lot_id"].map(prev_lot_dict))
            ].copy()

        # ✅ 이동한 Lot 중에서 IPLT 초과 위험이 높은 Lot 개수 체크
        moved_urgent_lots = moved_lots[moved_lots["lot_id"].isin(need_repair_lots["lot_id"])]
        moved_count = len(moved_urgent_lots)

        if len(need_repair_lots) == 0:
            return 0, 0, 0

        # ✅ 보상 계산 (이동한 위험 Lot의 비율 기반)
        urgent_lot_penaty = -min(moved_count / len(need_repair_lots), 1.0)/2
        return urgent_lot_penaty, moved_count, len(need_repair_lots)

    def format_log(self, array):
        # ✅ 출력 개수 설정
        split_sizes = [1, 24, 4, 8, 8]
        formatted_values = [f"{x:.4f}" for x in array]  # 소수점 4자리 포맷
        output_lines = []

        start = 0
        for size in split_sizes:
            end = start + size
            output_lines.append(", ".join(formatted_values[start:end]))  # 해당 개수만큼 잘라서 한 줄로
            start = end  # 다음 구간으로 이동

        return "\n".join(output_lines)  # 줄바꿈으로 조합