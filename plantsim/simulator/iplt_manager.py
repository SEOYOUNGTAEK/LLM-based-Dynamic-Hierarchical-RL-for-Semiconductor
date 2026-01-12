from pymongo import MongoClient

class IPLTManager:
    def __init__(self, iplts=None):
        client = MongoClient("mongodb://localhost:27017")  # MongoDB 연결
        db = client["plantsim"]
        self.collection = db["iplt"]
        self.iplts = iplts if iplts is not None else []

    def to_dict(self):
        # Lot 객체 리스트를 JSON 형식의 딕셔너리로 변환
        return {
            "device_id": [iplt["device_id"] for iplt in self.iplts],
            "iplt_time": [iplt["iplt_time"] for iplt in self.iplts],
        }

    def read_targets_from_db(self):
        iplts = self.collection.find()
        for iplt in iplts:
            self.iplts.append(iplt)

    def __repr__(self):
        return f"ipltManager(lots={self.iplts})"