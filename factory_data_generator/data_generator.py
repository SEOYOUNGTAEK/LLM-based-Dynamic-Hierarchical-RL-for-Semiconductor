import os
import sys
from factory_data_generator.generator.lot import Lot
from factory_data_generator.generator.eqp import Eqp
from factory_data_generator.generator.plan import Plan
from factory_data_generator.generator.target import Target
from factory_data_generator.generator.iplt import IPLT
from factory_data_generator.generator.db_storer import DBStorer
from logger import Logger

import random
from pymongo import MongoClient
class Generator:

    def __init__(self):
        self.logger = Logger().get_logger()

    def run(self):
        self.client = MongoClient("mongodb://localhost:27017")
        self.db_name = "plantsim"
        self.db = self.client[self.db_name]

        # 데이터베이스 삭제
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

        # 🚨 [수정] 총 긴급 Lot 목표 개수 설정
        URGENT_LOT_GOAL_PER_DEVICE = 10
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

                    # 🚨 [수정된 긴급 Lot 생성 로직]
                    is_eligible = (device_id == "A" or device_id == "B") and (process_id == 1)

                    if is_eligible and urgent_lots_created_total < URGENT_LOT_GOAL_TOTAL:

                        # A 제품 할당 조건 (10개 미만일 때만)
                        if device_id == "A" and urgent_a_count < URGENT_LOT_GOAL_PER_DEVICE:
                            current_priority = 1
                            urgent_a_count += 1
                            urgent_lots_created_total += 1
                            is_urgent = True

                        # B 제품 할당 조건 (10개 미만일 때만)
                        elif device_id == "B" and urgent_b_count < URGENT_LOT_GOAL_PER_DEVICE:
                            current_priority = 1
                            urgent_b_count += 1
                            urgent_lots_created_total += 1
                            is_urgent = True

                        if is_urgent:
                            # 긴급 Lot은 Process 1 (INPUT_BANK)으로 고정 투입
                            lot_process_id = 1
                            lot_buffer = buffers[1]
                            self.logger.info(f"URGENT LOT CREATED: ULot{unique_id} (Prio 1) for Device {device_id}")

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
        plan_manager.add_entity(IPLT("A", 25000))
        plan_manager.add_entity(IPLT("B", 20000))
        plan_manager.save_to_db()















