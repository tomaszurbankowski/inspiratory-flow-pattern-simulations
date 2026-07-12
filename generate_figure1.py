#!/usr/bin/env python3
"""
Generate Figure 1 for the manuscript:
"Effects of Inspiratory Waveform, Inspiratory Time, and Pause on
Compartmental Energy Routing in a Heterogeneous Two-Compartment
Respiratory Model"

Figure concept
--------------
A) Dynamic two-compartment model schematic.
B) Representative inspiratory flow partitioning over time.
C) Corresponding cumulative compartmental airway-level energy routing
   and the Partition Index.

The figure intentionally contains no overall title and no conventional legend.
The figure caption should be provided separately in the manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from mp_partitioning_v3_pattern import (
    Compartment,
    VentSettings,
    simulate_breath_two_compartments,
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "mathtext.default": "it",
    }
)

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
        -0.08,
        1.04,
        label,
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
    ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            fill=False,
            linewidth=LW_SCHEM,
            edgecolor=colour,
        )
    )

    x = np.linspace(x0 + 0.12 * w, x0 + 0.42 * w, 9)
    y_mid = y0 + 0.72 * h
    amp = 0.06 * h
    zig = np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]) * amp + y_mid
    ax.plot(x, zig, color=colour, lw=LW_SCHEM)
    ax.text(
        x0 + 0.27 * w,
        y0 + 0.85 * h,
        r_label,
        color=colour,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    xs = x0 + 0.72 * w
    ys = np.linspace(y0 + 0.20 * h, y0 + 0.72 * h, 9)
    x_offsets = np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]) * (0.035 * w)
    ax.plot(xs + x_offsets, ys, color=colour, lw=LW_SCHEM)
    ax.text(
        x0 + 0.82 * w,
        y0 + 0.48 * h,
        c_label,
        color=colour,
        ha="left",
        va="center",
        fontsize=10,
    )

    ax.plot(
        [x0, x0 + 0.12 * w],
        [y0 + 0.72 * h, y0 + 0.72 * h],
        color=colour,
        lw=LW_SCHEM,
    )
    ax.plot(
        [x0 + 0.42 * w, x0 + 0.72 * w],
        [y0 + 0.72 * h, y0 + 0.72 * h],
        color=colour,
        lw=LW_SCHEM,
    )
    ax.plot(
        [x0 + 0.72 * w, x0 + 0.72 * w],
        [y0 + 0.72 * h, y0 + 0.20 * h],
        color=colour,
        lw=LW_SCHEM,
    )

    ax.text(
        x0 + 0.50 * w,
        y0 + 0.18 * h,
        vol_label,
        color=colour,
        ha="center",
        va="center",
        fontsize=10,
    )

    ax.add_patch(
        FancyArrowPatch(
            (x0 - 0.10 * w, y0 + 0.72 * h),
            (x0, y0 + 0.72 * h),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=LW_SCHEM,
            color=colour,
        )
    )
    ax.text(
        x0 - 0.11 * w,
        y0 + 0.78 * h,
        flow_label,
        color=colour,
        ha="right",
        va="bottom",
        fontsize=10,
    )


def make_panel_a(ax) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "A")

    vent = Rectangle(
        (0.03, 0.38),
        0.13,
        0.24,
        fill=False,
        linewidth=LW_SCHEM,
        edgecolor="black",
    )
    ax.add_patch(vent)
    ax.text(0.095, 0.50, "VCV", ha="center", va="center", fontsize=11)
    ax.text(0.095, 0.35, r"$F_{tot}(t)$", ha="center", va="top", fontsize=10)

    ax.add_patch(
        FancyArrowPatch(
            (0.16, 0.50),
            (0.31, 0.50),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.8,
            color="black",
        )
    )

    node = Circle(
        (0.36, 0.50),
        0.040,
        fill=False,
        linewidth=LW_SCHEM,
        edgecolor="black",
    )
    ax.add_patch(node)
    ax.text(0.36, 0.50, r"$P_{aw}(t)$", ha="center", va="center", fontsize=8.5)
    ax.text(
        0.28,
        0.635,
        "airway opening",
        ha="center",
        va="center",
        fontsize=ANNOT_SIZE,
    )
    ax.text(0.36, 0.40, "PEEP", ha="center", va="top", fontsize=ANNOT_SIZE)

    ax.plot([0.400, 0.49], [0.50, 0.72], color="black", lw=LW_SCHEM)
    ax.plot([0.400, 0.49], [0.50, 0.28], color="black", lw=LW_SCHEM)

    draw_compartment(
        ax,
        x0=0.50,
        y0=0.58,
        w=0.34,
        h=0.24,
        r_label=r"$R_1$",
        c_label=r"$C_1$",
        flow_label=r"$F_1(t)$",
        vol_label=r"$V_1(t)$",
        colour=COL_C1,
    )
    draw_compartment(
        ax,
        x0=0.50,
        y0=0.10,
        w=0.34,
        h=0.24,
        r_label=r"$R_2$",
        c_label=r"$C_2$",
        flow_label=r"$F_2(t)$",
        vol_label=r"$V_2(t)$",
        colour=COL_C2,
    )

    ax.plot([0.49, 0.50], [0.72, 0.72], color="black", lw=LW_SCHEM)
    ax.plot([0.49, 0.50], [0.28, 0.28], color="black", lw=LW_SCHEM)

    ax.text(
        0.50,
        0.91,
        r"$F_{tot}(t)=F_1(t)+F_2(t)$",
        ha="left",
        va="center",
        fontsize=10,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def simulate_representative_case():
    """Return the representative case used in Figure 1."""
    vent = VentSettings(
        VT=0.50,
        Ti=1.0,
        RR=20.0,
        PEEP=5.0,
        dt=0.001,
        waveform="decelerating",
        pause_fraction=0.20,
    )
    comp1 = Compartment(C=0.04, R=8.0)
    comp2 = Compartment(C=0.08, R=4.0)
    result = simulate_breath_two_compartments(
        comp1,
        comp2,
        vent=vent,
        return_time_series=True,
    )
    return vent, result


def make_panel_b(ax, vent, res) -> None:
    add_panel_label(ax, "B")

    n_insp = res["n_insp"]
    t = res["t"][:n_insp]
    ftot = res["Ftot"][:n_insp]
    f1 = res["F1"][:n_insp]
    f2 = res["F2"][:n_insp]

    ax.plot(t, ftot, color=COL_FTOT, lw=LW_MAIN)
    ax.plot(t, f1, color=COL_C1, lw=LW_MAIN)
    ax.plot(t, f2, color=COL_C2, lw=LW_MAIN)

    flow_time = res["flow_time_s"]
    pause_time = res["pause_time_s"]
    if pause_time > 0:
        ax.axvspan(
            flow_time,
            flow_time + pause_time,
            color=COL_SHADE,
            alpha=0.15,
            lw=0,
        )
        ax.axvline(flow_time, color="0.5", ls="--", lw=LW_GUIDE)
        ax.text(
            flow_time + pause_time / 2,
            max(ftot) * 0.94,
            "pause",
            ha="center",
            va="top",
            fontsize=ANNOT_SIZE,
        )

    ax.text(
        t[np.argmax(ftot)] + 0.02,
        max(ftot) * 1.02,
        r"$F_{tot}$",
        color=COL_FTOT,
        ha="left",
        va="bottom",
        fontsize=10,
    )
    idx1 = min(int(0.30 / vent.dt), len(t) - 1)
    idx2 = min(int(0.58 / vent.dt), len(t) - 1)
    ax.text(
        t[idx1] + 0.02,
        f1[idx1] + 0.02,
        r"$F_1$",
        color=COL_C1,
        ha="left",
        va="bottom",
        fontsize=10,
    )
    ax.text(
        t[idx2] + 0.02,
        f2[idx2] - 0.03,
        r"$F_2$",
        color=COL_C2,
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.text(
        0.96,
        0.93,
        r"$C_2/C_1 = 2.0,\; R_2/R_1 = 0.5$",
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
    ymax = max(ftot.max(), f1.max(), f2.max())
    ax.set_ylim(min(0.0, f1.min(), f2.min()) - 0.04, ymax + 0.10)
    ax.set_xticks(np.arange(0, 1.01, 0.2))


def make_panel_c(ax, vent, res) -> None:
    add_panel_label(ax, "C")

    n_insp = res["n_insp"]
    t = res["t"][:n_insp]
    paw = res["Paw"][:n_insp]
    f1 = res["F1"][:n_insp]
    f2 = res["F2"][:n_insp]
    dp = paw - vent.PEEP

    power1 = dp * f1 * 0.098
    power2 = dp * f2 * 0.098
    energy1 = np.zeros_like(t)
    energy2 = np.zeros_like(t)
    energy1[1:] = np.cumsum((power1[:-1] + power1[1:]) * 0.5 * vent.dt)
    energy2[1:] = np.cumsum((power2[:-1] + power2[1:]) * 0.5 * vent.dt)
    energy_total = energy1 + energy2
    pi = energy2[-1] / energy_total[-1]

    ax.plot(t, energy_total, color=COL_FTOT, lw=LW_MAIN)
    ax.plot(t, energy1, color=COL_C1, lw=LW_MAIN)
    ax.plot(t, energy2, color=COL_C2, lw=LW_MAIN)

    flow_time = res["flow_time_s"]
    pause_time = res["pause_time_s"]
    if pause_time > 0:
        ax.axvspan(
            flow_time,
            flow_time + pause_time,
            color=COL_SHADE,
            alpha=0.15,
            lw=0,
        )
        ax.axvline(flow_time, color="0.5", ls="--", lw=LW_GUIDE)

    ax.text(
        t[-1] * 0.98,
        energy_total[-1],
        r"$E_{tot}$",
        color=COL_FTOT,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    ax.text(
        t[-1] * 0.98,
        energy1[-1] - 0.002,
        r"$E_1$",
        color=COL_C1,
        ha="right",
        va="top",
        fontsize=10,
    )
    ax.text(
        t[-1] * 0.98,
        energy2[-1] + 0.002,
        r"$E_2$",
        color=COL_C2,
        ha="right",
        va="bottom",
        fontsize=10,
    )

    ax.text(
        0.02,
        0.96,
        r"$E_i=\int (P_{aw}-PEEP)\,F_i(t)\,dt$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOT_SIZE,
    )
    ax.text(
        0.02,
        0.86,
        rf"$PI=E_2/E_{{tot}}={pi:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOT_SIZE,
    )

    ax.set_xlabel("Time during inspiration (s)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Cumulative energy (J)", fontsize=AXIS_LABEL_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, t[-1])
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(0, energy_total[-1] * 1.18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript Figure 1.")
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Optional output directory. By default, files are written to the "
            "directory containing this script."
        ),
    )
    parser.add_argument(
        "--basename",
        default="Figure_1",
        help="Base output filename without extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vent, result = simulate_representative_case()

    fig = plt.figure(figsize=(12.5, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0])

    make_panel_a(fig.add_subplot(grid[0, 0]))
    make_panel_b(fig.add_subplot(grid[0, 1]), vent, result)
    make_panel_c(fig.add_subplot(grid[0, 2]), vent, result)

    script_dir = Path(__file__).resolve().parent
    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else script_dir
    )
    outdir.mkdir(parents=True, exist_ok=True)
    out_base = outdir / args.basename

    fig.savefig(f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  {out_base}.png")
    print(f"  {out_base}.tiff")
    print(f"  {out_base}.svg")
    print(f"  {out_base}.pdf")


if __name__ == "__main__":
    main()
