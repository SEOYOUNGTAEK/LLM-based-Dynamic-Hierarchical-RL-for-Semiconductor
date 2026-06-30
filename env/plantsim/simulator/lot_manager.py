from pymongo import MongoClient

class LotManager:
    def __init__(self, lots=None):
        # 기본적으로 빈 리스트를 초기화하거나 제공된 Lot 객체들의 리스트로 초기화
        self.lots = lots if lots is not None else []
        client = MongoClient("mongodb://localhost:27017")  # MongoDB 연결
        db = client["plantsim"]  # 'plantsim' 데이터베이스
        self.collection = db["lot"]  # 'lot' 컬렉션

    def read_lots_from_db(self):
        lots = self.collection.find()
        for lot in lots:
            self.lots.append(lot)

    def add_lot(self, lot):
        self.lots.append(lot)

    def to_dict(self):
        # Lot 객체 리스트를 JSON 형식의 딕셔너리로 변환
        return {
            "lot_id": [lot["lot_id"] for lot in self.lots],
            "device_id": [lot["device_id"] for lot in self.lots],
            "process_id": [lot["process_id"] for lot in self.lots],
            "dest_step_tkout_time": [lot["dest_step_tkout_time"] for lot in self.lots],
            "mat_qty": [lot["mat_qty"] for lot in self.lots],
            "priority": [lot["priority"] for lot in self.lots],
            "current_location" :  [lot["current_location"] for lot in self.lots]
        }


















