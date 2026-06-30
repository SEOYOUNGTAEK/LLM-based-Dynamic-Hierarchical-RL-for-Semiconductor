# -*- coding: utf-8 -*-
"""
Generate fig3_sequence_improved.png
Original PlantUML structure re-built in matplotlib.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIG_W, FIG_H, DPI = 7.4, 5.0, 300
LOOP_H = 2.0

# ── participants ──────────────────────────────────────────────────────────────
PARTS = ["L2", "L1", "L0", "DT"]
PX    = {"L2": 13, "L1": 33, "L0": 59, "DT": 84}
PCLR  = {"L2": "#1F3F6E", "L1": "#2D6A2D", "L0": "#7A1F1F", "DT": "#4A3060"}
PLBL  = {
    "L2": "L2: LLM Strategist\n(per episode)",
    "L1": "L1: Safety Agent\n(per step)",
    "L0": "L0: DRL Agent\n(per step)",
    "DT": "Digital Twin\n(Environment)",
}
HDR_TOP = 97.0
HDR_BOT = 91.5
LINE_BOT =  2.0

# ── section blocks (each gets a labelled band across the top) ─────────────────
# (label, y_top, y_bot, band_fill, text_color)
# The label is drawn inside a 2-unit tall strip at the TOP of each band.
SECTIONS = [
    ("A.  Strategy Initialization  —  Low-Frequency, per episode",
     91.5, 74.5, "#FFF3E0", "#B06010"),
    ("B.  DRL Execution Loop  —  High-Frequency, per step ×N",
     74.5, 36.5, "#F0FFF0", "#2D6A2D"),
    ("C.  Post-Episode Update  —  Low-Frequency",
     36.5,  2.0, "#EEF2FF", "#1F3F6E"),
]

# Header strip height inside each section
SEC_HDR = 2.5

SIDEBAR = [
    ("A\nInit",      91.5, 74.5, "#B06010"),
    ("B\nDRL\nLoop", 74.5, 36.5, "#2D6A2D"),
    ("C\nEval",      36.5,  2.0, "#1F3F6E"),
]

# Events: placed BELOW the 2.5-unit header strip of each section
# Section A content: 91.5-2.5=89.0 → events at 86.5, 83.0, 79.5
# Section B content: 74.5-2.5=72.0 → events at 69.5, 65.5, 61.5, 57.5, 54.0, 49.5, 45.5, 40.0
# Section C content: 36.5-2.5=34.0 → events at 31.0, 27.5, 23.0, 15.0
EVENTS = [
    # ── A ──────────────────────────────────────────────────────────────────
    dict(y=86.5, src="DT", dst="L2",
         label="episode_summary",             clr=PCLR["DT"]),
    dict(y=83.0, src="L2", dst="L2",
         label="analyze_and_update_rules()",  clr=PCLR["L2"]),   # bot=81.0
    dict(y=79.0, src="L2", dst="L1",
         label="set_rules(new_rules)",         clr=PCLR["L2"]),
    # ── B ──────────────────────────────────────────────────────────────────
    dict(y=70.0, src="DT", dst="L1",
         label="get_state_summary()",          clr=PCLR["DT"]),
    dict(y=66.0, src="L1", dst="L0",
         label="get_action_mask_with_reason()", clr=PCLR["L1"]),
    dict(y=62.0, src="L1", dst="L0",
         label="action_mask",                  clr=PCLR["L1"]),
    dict(y=58.0, src="DT", dst="L0",
         label="state  (21-dim vector)",       clr=PCLR["DT"]),
    dict(y=54.5, src="L0", dst="L0",
         label="select_action(state, mask)",   clr=PCLR["L0"]),   # bot=52.5
    dict(y=49.5, src="L0", dst="DT",
         label="execute action  aₜ",           clr=PCLR["L0"]),
    dict(y=45.5, src="DT", dst="L0",
         label="next_state,  reward  R",       clr=PCLR["DT"]),
    dict(y=40.5, src="L0", dst="L0",
         label="replay_buffer.add()  →  update()", clr=PCLR["L0"], dashed=True),  # bot=38.5
    # ── C ──────────────────────────────────────────────────────────────────
    dict(y=33.0, src="DT", dst="L2",
         label="episode KPIs",                 clr=PCLR["DT"]),
    dict(y=29.0, src="L2", dst="L2",
         label="[train]  LLM analyze  →  revise rules", clr=PCLR["L2"]),  # bot=27.0
    dict(y=24.5, src="L2", dst="L1",
         label="updated rules  →  L1",         clr=PCLR["L2"]),
    dict(y=16.5, src="L2", dst="DT",
         label="[eval]  L2 rules frozen  —  no update", clr="#888888", dashed=True),
]


def draw_sections(ax):
    for label, yt, yb, fill, tclr in SECTIONS:
        # background fill
        ax.fill_between([4.5, 96.5], [yb]*2, [yt]*2, color=fill, zorder=0)
        # header strip at top of section
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
            ha="center", va="center", fontsize=6.4, fontweight="bold",
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

    ax.text(50, 99.3, "LLM-HRL Cross-Layer Data Sequence (Improved)",
            ha="center", va="top", fontsize=8.5, fontweight="bold",
            color="#222222", zorder=6)

    draw_sections(ax)

    for lbl, yt, yb, clr in SIDEBAR:
        draw_sidebar(ax, lbl, yt, yb, clr)

    for key in PARTS:
        draw_header(ax, key)

    for ev in EVENTS:
        draw_event(ax, ev)

    plt.tight_layout(pad=0.1)
    out = r"E:\SYT\SID_1031\SID_LLM_1031\iplt\paper_hicss\fig3_sequence_improved"
    fig.savefig(out+".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out+".pdf", bbox_inches="tight")
    print("Saved", out)
    plt.close(fig)

if __name__ == "__main__":
    main()
