import random
from pymongo import MongoClient
import numpy as np
class PlanManager:
    def __init__(self, plans=None):
        client = MongoClient("mongodb://localhost:27017")  # MongoDB 연결
        db = client["plantsim"]  # 'plantsim' 데이터베이스
        self.collection = db["plan"]  # 'lot' 컬렉션
        self.plans = plans if plans is not None else []
        self.lots = []

    def add_plan(self, plan):
        self.plans.append(plan)

    def generate_plan_lot(self):
        self.lots = []
        lot_id = 0  # 고유 Lot ID
        total_mat_qty = sum(plan["mat_qty"] for plan in self.plans)  # 전체 mat_qty 합산

        # ✅ 디바이스별 시간별 Lot 할당량을 저장하는 Dictionary
        device_lots_per_hour = {plan["device_id"]: [] for plan in self.plans}

        preferred_hours = {
            "A": (3, 8),  # (주요 피크, 서브 피크)
            "B": (8, 14),
            "C": (5, 20),
            "D": (6, 16),
        }

        for plan in self.plans:
            device_id = plan["device_id"]

            # ✅ 각 디바이스가 매 시간 일정한 비율로 Lot을 생성하도록 설정
            base_lots_per_hour = int((plan["mat_qty"] / total_mat_qty) * 30)
            base_lots_per_hour = max(10, min(base_lots_per_hour, 30))

            for hour in range(24):
                peak_hour, secondary_peak = preferred_hours[device_id]
                # ✅ 시간대별 가중치 적용 (특정 시간대 몰리게 하기)
                peak_effect_1 = np.exp(-((hour - peak_hour) ** 2) / 4) * 6
                peak_effect_2 = np.exp(-((hour - secondary_peak) ** 2) / 6) * 2  # 보조 피크

                # ✅ 기본 생성량 * 몰림 효과 적용
                peak_effect = peak_effect_1 + peak_effect_2 + 3  # **기본 값 추가해서 11시 이후도 일정 생성**

                lots_in_hour = int(base_lots_per_hour * peak_effect)
                lots_in_hour = max(random.randint(5, 10), min(lots_in_hour, 70))  # **최소 Lot 개수를 10으로 설정**

                # ✅ 디바이스별 시간별 Lot 생성량 저장
                device_lots_per_hour[device_id].append(lots_in_hour)

        # ✅ 매 시간마다 모든 디바이스가 동시에 Lot을 생성하도록 함
        for hour in range(24):
            for plan in self.plans:
                device_id = plan["device_id"]
                lots_in_hour = device_lots_per_hour[device_id][hour]  # 해당 시간의 Lot 개수

                for _ in range(lots_in_hour):
                    if plan["mat_qty"] <= 0:
                        break  # 남은 Lot이 없으면 중단

                    lot_mat_qty = min(plan["mat_qty"], 20)  # 한 번에 20씩 처리

                    # ✅ 1시간 내에서 랜덤하게 Lot 도착 시간 배분
                    arrive_time = hour * 3600 + int(np.random.normal(loc=1800, scale=600))
                    arrive_time = max(hour * 3600 + 1, min(arrive_time, (hour + 1) * 3600 - 1))

                    lot = {
                        "lot_id": f"VLot{lot_id}",
                        "device_id": device_id,
                        "arrive_time": arrive_time,
                        "dest_step_tkout_time": arrive_time - 1000,
                        "mat_qty": lot_mat_qty,
                    }
                    self.lots.append(lot)

                    plan["mat_qty"] -= lot_mat_qty  # 남은 Lot 수량 감소
                    lot_id += 1

        self.lots.sort(key=lambda x: x["arrive_time"])  # 도착 시간 기준 정렬

    def to_dict(self):
        self.generate_plan_lot()
        # Lot 데이터를 JSON 형식의 딕셔너리로 변환
        return {
            "lot_id": [lot["lot_id"] for lot in self.lots],
            "device_id": [lot["device_id"] for lot in self.lots],
            "dest_step_tkout_time": [lot["dest_step_tkout_time"] for lot in self.lots],
            "mat_qty": [lot["mat_qty"] for lot in self.lots],
            "arrive_time": [lot["arrive_time"] for lot in self.lots],
        }

    def read_plans_from_db(self):
        plans = self.collection.find()
        for plan in plans:
            self.plans.append(plan)

    def __repr__(self):
        return f"PlanManager(plans={self.plans})"
