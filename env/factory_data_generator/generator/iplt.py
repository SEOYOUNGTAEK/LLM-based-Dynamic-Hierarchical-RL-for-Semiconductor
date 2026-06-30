class IPLT:
    def __init__(self, device_id, iplt_time):
        self.device_id = device_id
        self.iplt_time = iplt_time

    def to_dict(self):
        return {
            "device_id": self.device_id,
            "iplt_time": self.iplt_time,
        }
