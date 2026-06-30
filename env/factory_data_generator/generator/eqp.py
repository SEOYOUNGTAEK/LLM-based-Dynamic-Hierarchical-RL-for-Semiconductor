class Eqp:
    def __init__(self, eqp_id, device_id, proc_time):
        self.eqp_id = eqp_id
        self.device_id = device_id
        self.proc_time = proc_time
        self.process_id = self.assign_process_id()  # process_id 자동 할당

    def assign_process_id(self):
        if self.eqp_id.startswith("PA"):
            return 1
        elif self.eqp_id.startswith("PB"):
            return 2
        elif self.eqp_id.startswith("PC"):
            return 3
        elif self.eqp_id.startswith("PD"):
            return 4
        elif self.eqp_id.startswith("PE"):
            return 5
        elif self.eqp_id.startswith("PF"):
            return 6
        else:
            return None  # Unknown process_id

    def to_dict(self):
        return {
            "eqp_id": self.eqp_id,
            "device_id": self.device_id,
            "proc_time": self.proc_time,
            "process_id": self.process_id,
        }
