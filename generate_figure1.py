#!/usr/bin/env python3
"""
Generate Figure 1 for the manuscript:
"Global Mechanical Power Does Not Capture Regional Energy Distribution:
Effects of Inspiratory Flow Pattern and Timing in Heterogeneous Lungs"

Figure concept:
A) Dynamic two-compartment model schematic.
B) Representative inspiratory flow partitioning over time in a heterogeneous case.
C) Corresponding cumulative inspiratory energy routing and Partition Index.

The figure intentionally contains NO title and NO legend.
Captions should be added separately in the manuscript.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

from mp_partitioning_v3_pattern import VentSettings, Compartment, simulate_breath_two_compartments


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "mathtext.default": "it",
})

PANEL_LABEL_SIZE = 13
AXIS_LABEL_SIZE = 10
ANNOT_SIZE = 8.5
COL_FTOT = "black"
COL_C1 = "#1f77b4"
COL_C2 = "#d62728"
COL_SHADE = "#9e9e9e"
LW_MAIN = 1.6
LW_SCHEM = 1.4
LW_GUIDE = 1.0


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08, 1.04, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_compartment(
    ax,
    x0: float,
    y0: float,
    w: float,
    h: float,
    r_label: str,
    c_label: str,
    flow_label: str,
    vol_label: str,
    colour: str,
) -> None:
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=LW_SCHEM, edgecolor=colour))

    x = np.linspace(x0 + 0.12 * w, x0 + 0.42 * w, 9)
    y_mid = y0 + 0.72 * h
    amp = 0.06 * h
    zig = np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]) * amp + y_mid
    ax.plot(x, zig, color=colour, lw=LW_SCHEM)
    ax.text(x0 + 0.27 * w, y0 + 0.85 * h, r_label, color=colour, ha="center", va="bottom", fontsize=10)

    xs = x0 + 0.72 * w
    ys = np.linspace(y0 + 0.20 * h, y0 + 0.72 * h, 9)
    x_offsets = np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]) * (0.035 * w)
    ax.plot(xs + x_offsets, ys, color=colour, lw=LW_SCHEM)
    ax.text(x0 + 0.82 * w, y0 + 0.48 * h, c_label, color=colour, ha="left", va="center", fontsize=10)

    ax.plot([x0, x0 + 0.12 * w], [y0 + 0.72 * h, y0 + 0.72 * h], color=colour, lw=LW_SCHEM)
    ax.plot([x0 + 0.42 * w, x0 + 0.72 * w], [y0 + 0.72 * h, y0 + 0.72 * h], color=colour, lw=LW_SCHEM)
    ax.plot([x0 + 0.72 * w, x0 + 0.72 * w], [y0 + 0.72 * h, y0 + 0.20 * h], color=colour, lw=LW_SCHEM)

    ax.text(x0 + 0.50 * w, y0 + 0.18 * h, vol_label, color=colour, ha="center", va="center", fontsize=10)

    ax.add_patch(FancyArrowPatch(
        (x0 - 0.10 * w, y0 + 0.72 * h),
        (x0, y0 + 0.72 * h),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=LW_SCHEM,
        color=colour,
    ))
    ax.text(x0 - 0.11 * w, y0 + 0.78 * h, flow_label, color=colour, ha="right", va="bottom", fontsize=10)



def make_panel_a(ax) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "A")

    vent = Rectangle((0.03, 0.38), 0.13, 0.24, fill=False, linewidth=LW_SCHEM, edgecolor="black")
    ax.add_patch(vent)
    ax.text(0.095, 0.50, "VCV", ha="center", va="center", fontsize=11)
    ax.text(0.095, 0.35, r"$F_{tot}(t)$", ha="center", va="top", fontsize=10)

    ax.add_patch(FancyArrowPatch(
        (0.16, 0.50), (0.31, 0.50),
        arrowstyle="-|>", mutation_scale=13,
        linewidth=1.8, color="black"
    ))

    node = Circle((0.36, 0.50), 0.040, fill=False, linewidth=LW_SCHEM, edgecolor="black")
    ax.add_patch(node)
    ax.text(0.36, 0.50, r"$P_{aw}(t)$", ha="center", va="center", fontsize=8.5)
    ax.text(0.28, 0.635, "airway opening", ha="center", va="center", fontsize=ANNOT_SIZE)
    ax.text(0.36, 0.40, r"PEEP", ha="center", va="top", fontsize=ANNOT_SIZE)

    ax.plot([0.400, 0.49], [0.50, 0.72], color="black", lw=LW_SCHEM)
    ax.plot([0.400, 0.49], [0.50, 0.28], color="black", lw=LW_SCHEM)

    draw_compartment(
        ax, x0=0.50, y0=0.58, w=0.34, h=0.24,
        r_label=r"$R_1$", c_label=r"$C_1$", flow_label=r"$F_1(t)$",
        vol_label=r"$V_1(t)$", colour=COL_C1,
    )
    draw_compartment(
        ax, x0=0.50, y0=0.10, w=0.34, h=0.24,
        r_label=r"$R_2$", c_label=r"$C_2$", flow_label=r"$F_2(t)$",
        vol_label=r"$V_2(t)$", colour=COL_C2,
    )

    ax.plot([0.49, 0.50], [0.72, 0.72], color="black", lw=LW_SCHEM)
    ax.plot([0.49, 0.50], [0.28, 0.28], color="black", lw=LW_SCHEM)

    ax.text(0.50, 0.91, r"$F_{tot}(t)=F_1(t)+F_2(t)$", ha="left", va="center", fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)



def simulate_representative_case():
    vent = VentSettings(VT=0.50, Ti=1.0, RR=20.0, PEEP=5.0, dt=0.001,
                        waveform="decelerating", pause_fraction=0.20)
    c1 = Compartment(C=0.04, R=8.0)
    c2 = Compartment(C=0.08, R=4.0)
    return vent, simulate_breath_two_compartments(c1, c2, vent=vent, return_time_series=True)



def make_panel_b(ax, vent, res) -> None:
    add_panel_label(ax, "B")

    n_insp = res["n_insp"]
    t = res["t"][:n_insp]
    Ftot = res["Ftot"][:n_insp]
    F1 = res["F1"][:n_insp]
    F2 = res["F2"][:n_insp]

    ax.plot(t, Ftot, color=COL_FTOT, lw=LW_MAIN)
    ax.plot(t, F1, color=COL_C1, lw=LW_MAIN)
    ax.plot(t, F2, color=COL_C2, lw=LW_MAIN)

    flow_time = res["flow_time_s"]
    pause_time = res["pause_time_s"]
    if pause_time > 0:
        ax.axvspan(flow_time, flow_time + pause_time, color=COL_SHADE, alpha=0.15, lw=0)
        ax.text(flow_time + pause_time / 2, max(Ftot) * 0.94, "pause",
                ha="center", va="top", fontsize=ANNOT_SIZE)

    ax.axvline(0.20, color="0.5", ls="--", lw=LW_GUIDE)
    ax.axvline(0.70, color="0.5", ls="--", lw=LW_GUIDE)

    ax.text(t[np.argmax(Ftot)] + 0.02, max(Ftot) * 1.02, r"$F_{tot}$", color=COL_FTOT,
            ha="left", va="bottom", fontsize=10)
    idx1 = min(int(0.30 / vent.dt), len(t) - 1)
    idx2 = min(int(0.58 / vent.dt), len(t) - 1)
    ax.text(t[idx1] + 0.02, F1[idx1] + 0.02, r"$F_1$", color=COL_C1, ha="left", va="bottom", fontsize=10)
    ax.text(t[idx2] + 0.02, F2[idx2] - 0.03, r"$F_2$", color=COL_C2, ha="left", va="top", fontsize=10)

    ax.text(
        0.96, 0.93, r"$C_2/C_1 = 2.0,\; R_2/R_1 = 0.5$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOT_SIZE,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        zorder=5,
    )

    ax.set_xlabel("Time during inspiration (s)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Flow (L/s)", fontsize=AXIS_LABEL_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, t[-1])
    ymax = max(Ftot.max(), F1.max(), F2.max())
    ax.set_ylim(min(0, F1.min(), F2.min()) - 0.04, ymax + 0.10)
    ax.set_xticks(np.arange(0, 1.01, 0.2))



def make_panel_c(ax, vent, res) -> None:
    add_panel_label(ax, "C")

    n_insp = res["n_insp"]
    t = res["t"][:n_insp]
    Paw = res["Paw"][:n_insp]
    F1 = res["F1"][:n_insp]
    F2 = res["F2"][:n_insp]
    dP = Paw - vent.PEEP

    p1 = dP * F1 * 0.098
    p2 = dP * F2 * 0.098
    E1 = np.zeros_like(t)
    E2 = np.zeros_like(t)
    dt = vent.dt
    E1[1:] = np.cumsum((p1[:-1] + p1[1:]) * 0.5 * dt)
    E2[1:] = np.cumsum((p2[:-1] + p2[1:]) * 0.5 * dt)
    Et = E1 + E2
    PI = E2[-1] / Et[-1]

    ax.plot(t, Et, color=COL_FTOT, lw=LW_MAIN)
    ax.plot(t, E1, color=COL_C1, lw=LW_MAIN)
    ax.plot(t, E2, color=COL_C2, lw=LW_MAIN)

    flow_time = res["flow_time_s"]
    pause_time = res["pause_time_s"]
    if pause_time > 0:
        ax.axvspan(flow_time, flow_time + pause_time, color=COL_SHADE, alpha=0.15, lw=0)
    ax.axvline(flow_time, color="0.5", ls="--", lw=LW_GUIDE)

    ax.text(t[-1] * 0.98, Et[-1], r"$E_{tot}$", color=COL_FTOT, ha="right", va="bottom", fontsize=10)
    ax.text(t[-1] * 0.98, E1[-1] - 0.002, r"$E_1$", color=COL_C1, ha="right", va="top", fontsize=10)
    ax.text(t[-1] * 0.98, E2[-1] + 0.002, r"$E_2$", color=COL_C2, ha="right", va="bottom", fontsize=10)

    ax.text(0.02, 0.96,
            r"$E_i=\int (P_{aw}-PEEP)\cdot F_i(t)\,dt$",
            transform=ax.transAxes, ha="left", va="top", fontsize=ANNOT_SIZE)
    ax.text(0.02, 0.86,
            rf"$PI=E_2/E_{{tot}}={PI:.2f}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=ANNOT_SIZE)

    ax.set_xlabel("Time during inspiration (s)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Cumulative energy (J)", fontsize=AXIS_LABEL_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, t[-1])
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(0, Et[-1] * 1.18)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 1 locally.")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--basename",
        default="Figure_1",
        help="Base filename without extension.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    vent, res = simulate_representative_case()

    fig = plt.figure(figsize=(12.5, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0])

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    make_panel_a(axA)
    make_panel_b(axB, vent, res)
    make_panel_c(axC, vent, res)

    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else script_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out_base = outdir / args.basename

    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  {out_base}.png")
    print(f"  {out_base}.svg")
    print(f"  {out_base}.pdf")


if __name__ == "__main__":
    main()
