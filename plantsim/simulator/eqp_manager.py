from pymongo import MongoClient

class EqpManager:
    def __init__(self, eqps=None):
        # 기본적으로 빈 리스트를 초기화하거나 제공된 Lot 객체들의 리스트로 초기화
        self.eqps = eqps if eqps is not None else []
        self.eqp_dict = {}  # eqp_id를 key, device_id를 value로 가지는 딕셔너리
        client = MongoClient("mongodb://localhost:27017")  # MongoDB 연결
        db = client["plantsim"]  # 'plantsim' 데이터베이스
        self.collection = db["eqp"]  # 'eqp' 컬렉션

    def add_eqp(self, eqp):
        self.eqps.append(eqp)
        self.eqp_dict[eqp["eqp_id"]] = eqp["device_id"]

    def to_dict(self):
        # Lot 객체 리스트를 JSON 형식의 딕셔너리로 변환
        return {
            "eqp_id": [eqp["eqp_id"] for eqp in self.eqps],
            "device_id": [eqp["device_id"] for eqp in self.eqps],
            "proc_time": [eqp["proc_time"] for eqp in self.eqps],
            "process_id": [eqp["process_id"] for eqp in self.eqps],
        }

    def read_eqps_from_db(self):
        eqps = self.collection.find()
        for eqp in eqps:
            self.eqps.append(eqp)
            self.eqp_dict[eqp["eqp_id"]] = eqp["device_id"]

    def get_total_equipment(self, process_id=None):
        if process_id is not None:
            filtered_eqps = [eqp for eqp in self.eqps if eqp.get("process_id") == process_id]
            return len(filtered_eqps)
        return len(self.eqps)

    def allocate_process_id(self, process_id, allocation):
        # Process ID가 일치하는 장비를 가져오기
        eqps_for_process = [eqp for eqp in self.eqps if eqp.get("process_id") == process_id]
        total_required = sum(allocation.values())
        total_available = len(eqps_for_process)

        if total_available < total_required:
            print(
                f"Warning: Not enough equipment for allocation! Required: {total_required}, Available: {total_available}")

        allocated_eqps = {}
        eqp_index = 0

        for device_id, count in allocation.items():
            for _ in range(count):
                if eqp_index < len(eqps_for_process):
                    eqp = eqps_for_process[eqp_index]
                    eqp["device_id"] = device_id  # 장비의 device_id 업데이트
                    allocated_eqps[eqp.get("eqp_id")] = device_id
                    eqp_index += 1
                else:
                    # 장비 부족 로그
                    #print(f"Insufficient equipment for device_id={device_id} (Needed: {count}, Allocated: {_})")
                    break

        return allocated_eqps
