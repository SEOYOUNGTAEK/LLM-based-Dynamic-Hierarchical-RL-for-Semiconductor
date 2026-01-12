from pymongo import MongoClient

class TargetManager:
    def __init__(self, targets=None):
        client = MongoClient("mongodb://localhost:27017")  # MongoDB 연결
        db = client["plantsim"]  # 'plantsim' 데이터베이스
        self.collection = db["target"]  # 'lot' 컬렉션
        # 기본적으로 빈 리스트를 초기화하거나 제공된 Lot 객체들의 리스트로 초기화
        self.targets = targets if targets is not None else []

    def add_target(self, target):
        # Lot 객체를 리스트에 추가
        self.targets.append(target)

    def to_dict(self):
        # Lot 객체 리스트를 JSON 형식의 딕셔너리로 변환
        return {
            "device_id": [target["device_id"] for target in self.targets],
            "mat_qty": [target["mat_qty"] for target in self.targets],
        }

    def read_targets_from_db(self):
        targets = self.collection.find()
        for target in targets:
            self.targets.append(target)

    def __repr__(self):
        return f"targetManager(lots={self.targets})"