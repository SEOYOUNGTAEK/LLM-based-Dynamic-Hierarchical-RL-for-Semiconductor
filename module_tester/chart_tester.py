import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
import pickle
from result_reporter.chart_generator import ChartGenerator

version_no = datetime.now().strftime('%Y%m%d_%H%M%S')
folder_name = f"output/{version_no}"
log_dir = "./" + folder_name
plot_dir = log_dir + '/plot/'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
if not os.path.exists(folder_name + f"/pickle"):
    os.makedirs(folder_name + f"/pickle")
if not os.path.exists(folder_name + f"/csv"):
    os.makedirs(folder_name + f"/csv")
if not os.path.exists(folder_name + f"/log"):
    os.makedirs(folder_name + f"/log")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

chart_generator = ChartGenerator(plot_dir)

with open("../pickle/data_20250205_135730/lot_history.pkl", "rb") as f:
    loaded_data = pickle.load(f)

chart_generator.draw_lot_chart(loaded_data)

print('plot_location : module_tester - '+plot_dir)

























