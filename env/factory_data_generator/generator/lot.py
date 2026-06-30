class Lot:
    def __init__(self, lot_id, device_id, process_id, dest_step_tkout_time, mat_qty, priority, current_location):
        self.lot_id = lot_id
        self.device_id = device_id
        self.process_id = process_id
        self.dest_step_tkout_time = dest_step_tkout_time
        self.mat_qty = mat_qty
        self.priority = priority
        self.current_location = current_location

    def to_dict(self):
        return {
            "lot_id": self.lot_id,
            "device_id": self.device_id,
            "process_id": self.process_id,
            "dest_step_tkout_time": self.dest_step_tkout_time,
            "mat_qty": self.mat_qty,
            "priority": self.priority,
            "current_location" : self.current_location,
        }
