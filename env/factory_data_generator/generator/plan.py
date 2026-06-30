class Plan:
    def __init__(self, device_id, mat_qty):
        self.device_id = device_id
        self.mat_qty = mat_qty

    def to_dict(self):
        return {
            "device_id": self.device_id,
            "mat_qty": self.mat_qty,
        }
