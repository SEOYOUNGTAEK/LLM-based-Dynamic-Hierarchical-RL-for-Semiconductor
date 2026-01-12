import json
from plantsim.simulator.lot_manager import LotManager
from plantsim.simulator.eqp_manager import EqpManager

lot_manager = LotManager()
lot_manager.read_lots_from_db()
json_data = json.dumps(lot_manager.to_dict())

print(json_data)

eqp_manager = EqpManager()
eqp_manager.read_eqps_from_db()
json_data = json.dumps(eqp_manager.to_dict())

print(json_data)
