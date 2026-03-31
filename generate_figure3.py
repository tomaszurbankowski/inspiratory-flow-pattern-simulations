#!/usr/bin/env python3
"""
generate_figure3.py

Journal-clean version of Figure 3:
- no title
- no legend
- reduced in-panel text
- no vertical pause marker
- derived directly from the study sweep outputs and model code
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mp_partitioning_v3_pattern import VentSettings, _build_inspiratory_flow_profile  # noqa: E402


CSV_PATH = BASE_DIR / "sweep_PI_MP_patterns.csv"
PNG_PATH = BASE_DIR / "Figure_3.png"
TIFF_PATH = BASE_DIR / "Figure_3.tiff"


def normalized_profile(waveform: str, ti: float = 1.0, pause_fraction: float = 0.0, dt: float = 0.001):
    vent = VentSettings(
        VT=0.50,
        Ti=ti,
        RR=20.0,
        PEEP=5.0,
        dt=dt,
        waveform=waveform,
        pause_fraction=pause_fraction,
    )
    n_cycle = int(np.round((60.0 / vent.RR) / vent.dt)) + 1
    ftot, n_flow, n_insp, n_pause = _build_inspiratory_flow_profile(vent, n_cycle)

    insp = ftot[:n_insp].copy()
    if np.max(insp) > 0:
        insp = insp / np.max(insp)

    t = np.arange(n_insp, dtype=float) * dt
    return t, insp


def main():
    df = pd.read_csv(CSV_PATH)

    pi_range = (
        df.groupby(["C2_over_C1", "R2_over_R1"])["PI"]
        .agg(["min", "max"])
        .reset_index()
    )
    pi_range["PI_range"] = pi_range["max"] - pi_range["min"]

    c_vals = np.sort(pi_range["C2_over_C1"].unique())
    r_vals = np.sort(pi_range["R2_over_R1"].unique())

    z_range = (
        pi_range.pivot(index="R2_over_R1", columns="C2_over_C1", values="PI_range")
        .reindex(index=r_vals, columns=c_vals)
        .to_numpy()
    )

    waveform_order = ["square", "decelerating", "sinusoidal"]
    panel_labels = ["A", "B", "C"]
    waveform_labels = ["Square", "Decelerating", "Sinusoidal"]

    fig = plt.figure(figsize=(10.2, 5.7), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=4,
        width_ratios=[1.0, 1.0, 1.0, 2.25],
        height_ratios=[1.0, 1.0, 1.0],
    )

    for i, (wf, p_label, wf_label) in enumerate(zip(waveform_order, panel_labels, waveform_labels)):
        ax = fig.add_subplot(gs[i, 0:3])

        t0, y0 = normalized_profile(wf, ti=1.0, pause_fraction=0.0)
        t1, y1 = normalized_profile(wf, ti=1.0, pause_fraction=0.2)

        x0 = t0 / t0[-1]
        x1 = t1 / t1[-1]

        ax.plot(x0, y0, linewidth=1.8)
        ax.plot(x1, y1, linewidth=1.8, linestyle="--")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.05, 1.12)
        ax.set_yticks([0.0, 0.5, 1.0])

        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Inspiratory time (normalised)")

        ax.set_ylabel("Flow\n(normalised)")
        ax.text(0.01, 0.97, p_label, transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=12)
        ax.text(0.09, 0.97, wf_label, transform=ax.transAxes, va="top", ha="left", fontsize=11)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axh = fig.add_subplot(gs[:, 3])
    im = axh.imshow(
        z_range,
        origin="lower",
        aspect="auto",
        extent=[c_vals.min(), c_vals.max(), r_vals.min(), r_vals.max()],
    )

    axh.plot(1.0, 1.0, marker="o", markersize=4)
    axh.text(1.08, 1.08, "symmetry", fontsize=8)

    max_row = pi_range.loc[pi_range["PI_range"].idxmax()]
    axh.plot(max_row["C2_over_C1"], max_row["R2_over_R1"], marker="x", markersize=6)
    axh.text(
        max_row["C2_over_C1"] - 0.95,
        max_row["R2_over_R1"] - 0.12,
        f"ΔPI = {max_row['PI_range']:.3f}",
        fontsize=8,
    )

    axh.text(0.01, 0.99, "D", transform=axh.transAxes, va="top", ha="left", fontweight="bold", fontsize=12)
    axh.set_xlabel("C₂/C₁")
    axh.set_ylabel("R₂/R₁")

    cbar = fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)
    cbar.set_label("PI range")

    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(TIFF_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {TIFF_PATH}")


if __name__ == "__main__":
    main()
