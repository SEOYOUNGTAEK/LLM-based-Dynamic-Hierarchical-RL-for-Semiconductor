import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import json
import re
from collections import deque
import os
from analysis.result_reporter.paper_visualizer import draw_kpi_trend, draw_reward_convergence, draw_governance_activity


class ChartGenerator:

    def __init__(self, plot_dir, condition='Unknown', eval_start=70):
        self.plot_dir = plot_dir
        self.condition = condition
        self.eval_start = eval_start

    # ── Paper-quality charts (위임) ──────────────────────────────
    def draw_kpi_trend(self, all_results, episode):
        draw_kpi_trend(all_results, self.eval_start, self.plot_dir, episode, self.condition)

    def draw_reward_convergence(self, rewards, losses, episode):
        draw_reward_convergence(rewards, losses, self.eval_start, self.plot_dir, episode, self.condition)

    def draw_governance_activity(self, all_results, episode):
        draw_governance_activity(all_results, self.eval_start, self.plot_dir, episode, self.condition)
    def draw_eqp_chart(self, dic, episode=0):
        # 1️⃣ 잘못된 연속된 따옴표 제거
        fixed_string = re.sub(r',]', ']', dic)
        fixed_string = json.loads(fixed_string)
        # DataFrame 변환
        df = pd.DataFrame(fixed_string)
        unique_eqp = df["eqp_id"].unique()
        color_map = {device: plt.cm.get_cmap("tab10")(i) for i, device in enumerate(df["device_id"].unique())}

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, eqp in enumerate(unique_eqp):
            eqp_data = df[df["eqp_id"] == eqp]
            for j in range(len(eqp_data) - 1):
                start_time = eqp_data.iloc[j]["sim_time"]
                end_time = eqp_data.iloc[j + 1]["sim_time"]
                device = eqp_data.iloc[j]["device_id"]
                ax.barh(i, end_time - start_time, left=start_time, color=color_map[device], edgecolor="black")

        # 축 설정
        ax.set_yticks(np.arange(len(unique_eqp)))
        ax.set_yticklabels(unique_eqp)
        ax.set_xlabel("Simulation Time")
        ax.set_ylabel("Equipment ID")
        ax.set_title("Gantt Chart of Equipment Device Transitions")

        # 범례 추가
        legend_labels = [plt.Rectangle((0, 0), 1, 1, color=color_map[device]) for device in color_map]
        ax.legend(legend_labels, color_map.keys(), title="Device ID", loc="upper right")

        graph_filename = os.path.join(self.plot_dir, f"eqp_plot_{episode + 1}.png")
        plt.savefig(graph_filename)  # 파일로 저장!
        plt.show()


    def draw_lot_chart(self, data, episode=0):
        # 1️⃣ JSON 데이터를 DataFrame으로 변환
        fixed_string = json.loads(data)
        df = pd.DataFrame(fixed_string)

        # 2️⃣ Process 1~6만 사용
        df["process_id"] = df["process_id"].astype(int)
        df = df[df["process_id"].between(1, 7)]

        # 3️⃣ Lot 별 Process 변경 시간 기록
        df = df.sort_values(by=["lot_id", "sim_time"])
        lot_history = df.groupby("lot_id")[["sim_time", "process_id"]].agg(list)

        # 4️⃣ Process 별 Queue 생성 (1~6)
        process_queues = {p: deque() for p in range(1, 8)}

        # 5️⃣ Simulation Time 설정
        unique_sim_times = sorted(df["sim_time"].unique())
        sim_time_bins = list(range(0, unique_sim_times[-1] + 300, 300))

        # 6️⃣ WIP 기록
        wip_tracking = {t: {p: 0 for p in range(1, 8)} for t in sim_time_bins}

        # 7️⃣ Lot 이동을 추적하는 이벤트 기반 시뮬레이션
        lot_positions = {}
        events = []

        for lot, (times, processes) in lot_history.iterrows():
            for idx in range(len(times)):
                events.append((times[idx], lot, processes[idx]))

        events.sort()

        # ✅ 7-1: simtime이 비어있는 부분을 보완하기 위한 세트 생성
        event_sim_times = {t for t, _, _ in events}  # events에 존재하는 simtime들
        all_sim_times = set(sim_time_bins)  # 300 단위의 모든 simtime

        # ✅ 7-2: events에 없는 simtime을 추가 (직전 상태 유지)
        missing_sim_times = sorted(all_sim_times - event_sim_times)  # 빠진 시간 찾기
        for t in missing_sim_times:
            events.append((t, None, None))  # Lot 없음, Process 없음 -> 기존 값 유지

        # ✅ 7-3: 추가된 이벤트들을 다시 정렬
        events.sort()

        # 8️⃣ 이벤트 기반 처리
        last_wip = {p: 0 for p in range(1, 8)}  # 🔥 이전 WIP 값을 저장해서 유지할 때 사용

        for t, lot, process in events:
            if lot is not None:
                if lot in lot_positions:
                    prev_process = lot_positions[lot]
                    if lot in process_queues[prev_process]:  # 🔥 이전 공정에서 Lot 제거
                        process_queues[prev_process].remove(lot)

                process_queues[process].append(lot)
                lot_positions[lot] = process

            # ✅ 8-1: simtime이 없었던 부분은 이전 WIP 값을 유지
            closest_bin = max([b for b in sim_time_bins if b <= t])
            for p in range(1, 8):
                if lot is None:  # 🔥 simtime 보완용 값이라면 이전 값을 유지
                    wip_tracking[closest_bin][p] = last_wip[p]
                else:
                    wip_tracking[closest_bin][p] = len(process_queues[p])
                    last_wip[p] = len(process_queues[p])  # ✅ 최신 WIP 저장

        # 9️⃣ DataFrame 변환
        wip_df = pd.DataFrame.from_dict(wip_tracking, orient="index").fillna(0)

        # 🔟 Process 순서 정렬
        sorted_columns = sorted(wip_df.columns, key=lambda x: int(x))
        wip_df = wip_df[sorted_columns]

        # 📊 Stacked Bar Chart 그리기
        plt.figure(figsize=(12, 6))
        wip_df.plot(kind="bar", stacked=True, colormap="tab10", width=0.8)

        plt.xlabel("Simulation Time (Grouped by 300)")
        plt.ylabel("WIP Count")
        plt.title("WIP per Process by Simulation Time (Fixed & Interpolated)")
        plt.xticks(rotation=45)
        plt.legend(title="Process ID")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        graph_filename = os.path.join(self.plot_dir, f"lot_plot_{episode + 1}.png")
        plt.savefig(graph_filename)  # 파일로 저장!
