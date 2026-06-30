# -*- coding: utf-8 -*-
"""
Generate fig3_sequence_improved_ko.png  — Korean version
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# ── Korean font setup ─────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False   # minus sign fix

FIG_W, FIG_H, DPI = 7.4, 5.0, 300
LOOP_H = 2.0

# ── participants ──────────────────────────────────────────────────────────────
PARTS = ["L2", "L1", "L0", "DT"]
PX    = {"L2": 13, "L1": 33, "L0": 59, "DT": 84}
PCLR  = {"L2": "#1F3F6E", "L1": "#2D6A2D", "L0": "#7A1F1F", "DT": "#4A3060"}
PLBL  = {
    "L2": "L2: LLM 전략 에이전트\n(에피소드 단위)",
    "L1": "L1: 안전 에이전트\n(스텝 단위)",
    "L0": "L0: DRL 에이전트\n(스텝 단위)",
    "DT": "디지털 트윈\n(환경)",
}
HDR_TOP = 97.0
HDR_BOT = 91.5
LINE_BOT =  2.0

SECTIONS = [
    ("A.  전략 초기화  —  저빈도, 에피소드 단위",
     91.5, 74.5, "#FFF3E0", "#B06010"),
    ("B.  DRL 실행 루프  —  고빈도, 매 스텝 반복",
     74.5, 36.5, "#F0FFF0", "#2D6A2D"),
    ("C.  에피소드 종료 후 업데이트  —  저빈도",
     36.5,  2.0, "#EEF2FF", "#1F3F6E"),
]

SEC_HDR = 2.5

SIDEBAR = [
    ("A\n초기화",    91.5, 74.5, "#B06010"),
    ("B\nDRL\n루프", 74.5, 36.5, "#2D6A2D"),
    ("C\n평가",      36.5,  2.0, "#1F3F6E"),
]

EVENTS = [
    # ── A ──────────────────────────────────────────────────────────────────
    dict(y=86.5, src="DT", dst="L2",
         label="에피소드 요약 수신",             clr=PCLR["DT"]),
    dict(y=83.0, src="L2", dst="L2",
         label="analyze_and_update_rules()  실행",  clr=PCLR["L2"]),
    dict(y=79.0, src="L2", dst="L1",
         label="set_rules(new_rules)  — 새 룰 전달", clr=PCLR["L2"]),
    # ── B ──────────────────────────────────────────────────────────────────
    dict(y=70.0, src="DT", dst="L1",
         label="get_state_summary()  — 상태 요약 요청", clr=PCLR["DT"]),
    dict(y=66.0, src="L1", dst="L0",
         label="get_action_mask_with_reason()  — 마스크 생성", clr=PCLR["L1"]),
    dict(y=62.0, src="L1", dst="L0",
         label="action_mask  — 허용 행동 집합 전달",  clr=PCLR["L1"]),
    dict(y=58.0, src="DT", dst="L0",
         label="상태 벡터  s  (21차원)",              clr=PCLR["DT"]),
    dict(y=54.5, src="L0", dst="L0",
         label="select_action(state, mask)  — 행동 선택", clr=PCLR["L0"]),
    dict(y=49.5, src="L0", dst="DT",
         label="행동 실행  aₜ",                       clr=PCLR["L0"]),
    dict(y=45.5, src="DT", dst="L0",
         label="다음 상태  s',  보상  R  반환",        clr=PCLR["DT"]),
    dict(y=40.5, src="L0", dst="L0",
         label="replay_buffer.add()  →  update()  — 학습",
         clr=PCLR["L0"], dashed=True),
    # ── C ──────────────────────────────────────────────────────────────────
    dict(y=33.0, src="DT", dst="L2",
         label="에피소드 KPI 전송  (사이클타임, 생산달성률 등)", clr=PCLR["DT"]),
    dict(y=29.0, src="L2", dst="L2",
         label="[학습 시]  LLM 분석  →  룰 재생성",  clr=PCLR["L2"]),
    dict(y=24.5, src="L2", dst="L1",
         label="갱신된 룰  →  L1 전달",               clr=PCLR["L2"]),
    dict(y=16.5, src="L2", dst="DT",
         label="[평가 시]  L2 룰 동결  —  업데이트 없음", clr="#888888", dashed=True),
]


def draw_sections(ax):
    for label, yt, yb, fill, tclr in SECTIONS:
        ax.fill_between([4.5, 96.5], [yb]*2, [yt]*2, color=fill, zorder=0)
        strip_bot = yt - SEC_HDR
        ax.fill_between([4.5, 96.5], [strip_bot]*2, [yt]*2,
                        color=tclr, alpha=0.18, zorder=1)
        ax.plot([4.5, 96.5], [strip_bot]*2, color=tclr, lw=0.5, alpha=0.4, zorder=1)
        ax.text(50, yt - SEC_HDR/2, label,
                ha="center", va="center", fontsize=6.3, fontweight="bold",
                color=tclr, zorder=3)


def draw_sidebar(ax, label, yt, yb, clr):
    rect = mpatches.FancyBboxPatch(
        (0.3, yb+0.4), 3.6, yt-yb-0.8,
        boxstyle="round,pad=0.2",
        lw=0.7, edgecolor=clr, facecolor=clr+"18", zorder=3)
    ax.add_patch(rect)
    ax.text(2.1, (yt+yb)/2, label,
            ha="center", va="center", fontsize=5.8, fontweight="bold",
            color=clr, zorder=4)


def draw_header(ax, key):
    x, c = PX[key], PCLR[key]
    rect = mpatches.FancyBboxPatch(
        (x-7.5, HDR_BOT), 15, HDR_TOP-HDR_BOT,
        boxstyle="round,pad=0.3",
        lw=1.2, edgecolor=c, facecolor=c+"22", zorder=5)
    ax.add_patch(rect)
    ax.text(x, (HDR_TOP+HDR_BOT)/2, PLBL[key],
            ha="center", va="center", fontsize=6.2, fontweight="bold",
            color=c, zorder=6)
    ax.plot([x, x], [HDR_BOT, LINE_BOT],
            color=c, lw=0.55, ls="--", alpha=0.28, zorder=1)


def draw_event(ax, ev):
    src=ev["src"]; dst=ev["dst"]; y=ev["y"]
    clr=ev["clr"]; label=ev["label"]
    ls="--" if ev.get("dashed") else "-"

    if src == dst:
        x = PX[src]
        lw = 7.5
        xs = [x, x+lw, x+lw, x]
        ys = [y, y, y-LOOP_H, y-LOOP_H]
        ax.plot(xs, ys, color=clr, lw=0.85, ls=ls, zorder=3)
        ax.annotate("", xy=(x, y-LOOP_H), xytext=(x+0.3, y-LOOP_H),
                    arrowprops=dict(arrowstyle="-|>", color=clr,
                                   lw=0.85, mutation_scale=5), zorder=3)
        ax.text(x+lw+0.5, y-LOOP_H/2, label,
                ha="left", va="center", fontsize=5.9,
                color=clr, fontweight="bold", zorder=4)
    else:
        x0, x1 = PX[src], PX[dst]
        ax.plot(x0, y, "o", color=clr, ms=2.8, zorder=4)
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=clr,
                                   lw=0.9, mutation_scale=6.5,
                                   linestyle=ls),
                    zorder=3)
        xm = (x0+x1)/2
        ax.text(xm, y-0.55, label,
                ha="center", va="top", fontsize=6.0,
                color=clr, fontweight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.07",
                          fc="white", ec="none", alpha=0.88))


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 99.3, "LLM-HRL 계층 간 데이터 시퀀스 다이어그램",
            ha="center", va="top", fontsize=9.0, fontweight="bold",
            color="#222222", zorder=6)

    draw_sections(ax)

    for lbl, yt, yb, clr in SIDEBAR:
        draw_sidebar(ax, lbl, yt, yb, clr)

    for key in PARTS:
        draw_header(ax, key)

    for ev in EVENTS:
        draw_event(ax, ev)

    plt.tight_layout(pad=0.1)
    out = r"E:\SYT\SID_1031\SID_LLM_1031\iplt\paper_hicss\fig3_sequence_improved_ko"
    fig.savefig(out+".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out+".pdf", bbox_inches="tight")
    print("Saved", out)
    plt.close(fig)

if __name__ == "__main__":
    main()
