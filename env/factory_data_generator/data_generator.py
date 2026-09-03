import os
import sys
from factory_data_generator.generator.lot import Lot
from factory_data_generator.generator.eqp import Eqp
from factory_data_generator.generator.plan import Plan
from factory_data_generator.generator.target import Target
from factory_data_generator.generator.iplt import IPLT
from factory_data_generator.generator.db_storer import DBStorer
from utils.logger import Logger

import random
from pymongo import MongoClient
class Generator:

    def __init__(self):
        self.logger = Logger().get_logger()

    def run(self, urgent_lot_ratio: float = None, urgent_entry_type: str = "early"):
        """
        urgent_lot_ratio  : 긴급 Lot 비율 (0.0~1.0). None이면 기본값(7.7%) 사용.
        urgent_entry_type : 긴급 Lot 진입 유형
            "early"  - 투입형: 공정 초기(Process 1) 투입. 기본값/현행 조건.
            "mid"    - 전환형: 공정 중반(Process 3-4) 긴급 전환. 납기 위급 시나리오.
            "late"   - 막바지형: 공정 후반(Process 5-6) 투입. 잔여 공정 극소 시나리오.
            "mixed"  - 혼합형: early 절반 + late 절반. 실제 운영 혼재 시나리오.
        """
        self.urgent_lot_ratio = urgent_lot_ratio
        self.urgent_entry_type = urgent_entry_type
        self.client = MongoClient("mongodb://localhost:27017")
        self.db_name = "plantsim"
        self.db = self.client[self.db_name]

        self.client.drop_database(self.db_name)
        self.logger.info(f"'{self.db_name}' Database Deleted.")

        self.generate_plan()
        self.generate_target()
        self.generate_lot()
        self.generate_eqp()
        self.generate_iplt()
        self.logger.info(f"'{self.db_name}' Database Created.")

    def generate_lot(self):
        tat_values = {
            1: 0,
            2: -500,
            3: -1800,
            4: -2000,
            5: -3000,
            6: -3500
        }

        lot_counts = {
            "A": {1: 10, 2: 10, 3: 3, 4: 1, 5: 10, 6: 15},
            "B": {1: 5, 2: 15, 3: 15, 4: 15, 5: 15, 6: 1},
            "C": {1: 10, 2: 15, 3: 20, 4: 15, 5: 5, 6: 3},
            "D": {1: 20, 2: 15, 3: 15, 4: 5, 5: 15, 6: 5},
        }

        buffers = {
            1: "INPUT_BANK",
            2: "BufferB",
            3: "BufferC",
            4: "BufferD",
            5: "BufferE",
            6: "BufferF"
        }

        lot_manager = DBStorer("lot")
        unique_id = 0
        # ✅ (신규) 긴급 Lot 카운터 (총 20개를 추적)
        urgent_lots_created_total = 0
        urgent_a_count = 0
        urgent_b_count = 0

        # 긴급 Lot 비율 설정
        # 전체 Lot 수: A(49)+B(66)+C(68)+D(75) = 258
        TOTAL_LOTS = 258
        if self.urgent_lot_ratio is not None:
            # 비율에 따라 목표 개수 계산, A/B 각 절반씩 배분
            target_total = max(2, int(TOTAL_LOTS * self.urgent_lot_ratio))
            URGENT_LOT_GOAL_PER_DEVICE = target_total // 2
            URGENT_LOT_GOAL_TOTAL = URGENT_LOT_GOAL_PER_DEVICE * 2
            self.logger.info(f"Urgent lot ratio: {self.urgent_lot_ratio*100:.1f}% → {URGENT_LOT_GOAL_TOTAL} lots")
        else:
            URGENT_LOT_GOAL_PER_DEVICE = 10   # 기본값: 7.7%
            URGENT_LOT_GOAL_TOTAL = 20

        for device_id, processes in lot_counts.items():
            for process_id, count in processes.items():
                tat = tat_values[process_id]
                buffer = buffers[process_id]

                for _ in range(count):
                    current_priority = 5  # 기본값
                    is_urgent = False
                    lot_process_id = process_id  # 기본값
                    lot_buffer = buffer  # 기본값

                    # 긴급 Lot 진입 유형에 따른 eligible process 결정
                    # "early" : process 1 (현행)
                    # "mid"   : process 3~4
                    # "late"  : process 5~6
                    # "mixed" : A=early(process 1), B=late(process 5)
                    etype = getattr(self, "urgent_entry_type", "early")
                    if etype == "early":
                        eligible_process = {1}
                        urgent_proc = 1
                    elif etype == "mid":
                        eligible_process = {3, 4}
                        urgent_proc = 3
                    elif etype == "late":
                        eligible_process = {5, 6}
                        urgent_proc = 5
                    elif etype == "mixed":
                        # A → early(1), B → late(5)
                        if device_id == "A":
                            eligible_process = {1}
                            urgent_proc = 1
                        else:
                            eligible_process = {5, 6}
                            urgent_proc = 5
                    else:
                        eligible_process = {1}
                        urgent_proc = 1

                    is_eligible = (device_id == "A" or device_id == "B") and (process_id in eligible_process)

                    if is_eligible and urgent_lots_created_total < URGENT_LOT_GOAL_TOTAL:

                        if device_id == "A" and urgent_a_count < URGENT_LOT_GOAL_PER_DEVICE:
                            current_priority = 1
                            urgent_a_count += 1
                            urgent_lots_created_total += 1
                            is_urgent = True

                        elif device_id == "B" and urgent_b_count < URGENT_LOT_GOAL_PER_DEVICE:
                            current_priority = 1
                            urgent_b_count += 1
                            urgent_lots_created_total += 1
                            is_urgent = True

                        if is_urgent:
                            lot_process_id = urgent_proc
                            lot_buffer = buffers[urgent_proc]
                            self.logger.info(
                                f"URGENT LOT CREATED: ULot{unique_id} (Prio 1) "
                                f"Device={device_id} entry_type={etype} proc={urgent_proc}"
                            )

                    lot_id_prefix = "ULot" if is_urgent else "Lot"

                    lot_manager.add_entity(
                        Lot(
                            f"{lot_id_prefix}{unique_id}",
                            device_id,
                            lot_process_id,  # 🚨 [수정] 조정된 Process ID 사용
                            random.randint(-500, -250) + tat,
                            20,
                            current_priority,
                            lot_buffer  # 🚨 [수정] 조정된 Buffer 사용
                        )
                    )
                    unique_id += 1

        lot_manager.save_to_db()

    def generate_eqp(self):
        self.proc_time = {}
        self.proc_time["A"] = 125
        self.proc_time["B"] = 200
        self.proc_time["C"] = 380
        self.proc_time["D"] = 240
        self.proc_time["E"] = 445
        self.proc_time["F"] = 240

        eqp_manager = DBStorer("eqp")
        eqp_config = [
            ("PA", "*", 5, "A"),
            ("PB", "*", 7, "B"),
            ("PC", "*", 11, "C"),
            ("PD", "*", 8, "D"),
            ("PE", "A", 1, "E"),
            ("PE", "B", 2, "E"),
            ("PE", "C", 4, "E"),
            ("PE", "D", 5, "E"),
            ("PF", "*", 13, "F"),
        ]

        # 설비군별 카운터를 저장할 딕셔너리
        group_counters = {group: 1 for group, _, _, _ in eqp_config}

        # 설비 생성
        for prefix, target, count, proc_key in eqp_config:
            for _ in range(count):
                # 설비군별로 ID 증가
                eqp_id = group_counters[prefix]
                eqp_name = f"{prefix}_EQP{eqp_id}"
                group_counters[prefix] += 1  # 설비군별 카운터 증가

                # 설비 추가
                eqp_manager.add_entity(Eqp(eqp_name, target, self.proc_time[proc_key]))

        eqp_manager.save_to_db()

    def generate_plan(self):
        plan_manager = DBStorer("plan")
        plan_manager.add_entity(Plan("A", 1500000))
        plan_manager.add_entity(Plan("B", 1500000))
        plan_manager.add_entity(Plan("C", 1500000))
        plan_manager.add_entity(Plan("D", 1500000))
        plan_manager.save_to_db()

    def generate_target(self):
        target_manager = DBStorer("target")
        target_manager.add_entity(Target("A", 13000))
        target_manager.add_entity(Target("B", 14000))
        target_manager.add_entity(Target("C", 12000))
        target_manager.add_entity(Target("D", 12000))
        target_manager.save_to_db()


    def generate_iplt(self):
        plan_manager = DBStorer("iplt")
        # [수정] 긴급 자동 우선 제거: 긴급 device(A,B)의 IPLT를 매우 길게 고정.
        #   원래값(A=25000, B=20000)은 긴급 lot의 remain_iplt_time을 촉박하게 만들어,
        #   IPLT 기반 정렬(Action 6~8)이 긴급을 "자동으로" 앞 순위에 올렸다.
        #   → DRL이 Action 9를 안 써도 긴급이 처리되어 학습 과제가 성립 안 함.
        #   이제 IPLT를 시뮬레이션 길이(86400)보다 훨씬 크게 설정해 긴급이 절대
        #   IPLT-촉박이 되지 않게 한다. 그러면 긴급 우선은 오직 Action 9(긴급=-10)
        #   로만 가능 → DRL이 "긴급 우선 타이밍"을 의식적으로 학습해야 함.
        BIG_IPLT = 10_000_000  # 사실상 무한대(IPLT 초과/촉박 발생 안 함)
        plan_manager.add_entity(IPLT("A", BIG_IPLT))
        plan_manager.add_entity(IPLT("B", BIG_IPLT))
        plan_manager.save_to_db()
        self.logger.info(f"IPLT (urgent auto-priority disabled): A=B={BIG_IPLT}")















