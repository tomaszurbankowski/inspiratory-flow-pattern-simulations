#!/usr/bin/env python3
"""
generate_figure4.py

FINAL submission-ready Figure 4:
- no title
- no legend
- no orange points
- improved clarity and aesthetics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("sweep_PI_MP_patterns.csv")

    mp = df["MPtotal_Jmin"].to_numpy(dtype=float)
    pi = df["PI"].to_numpy(dtype=float)

    # ===== Panel B envelope =====
    bins = np.linspace(mp.min(), mp.max(), 120)
    centers = 0.5 * (bins[:-1] + bins[1:])

    pi_min = np.full(len(centers), np.nan)
    pi_max = np.full(len(centers), np.nan)

    for i in range(len(centers)):
        mask = (mp >= bins[i]) & (mp < bins[i + 1])
        if np.any(mask):
            pi_min[i] = np.min(pi[mask])
            pi_max[i] = np.max(pi[mask])

    # ===== Figure =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    mp_center = 6.5
    mp_halfwidth = 0.5
    band_alpha = 0.10

    # ===== Panel A =====
    ax = axes[0]

    # clean background (only blue cloud)
    ax.scatter(mp, pi, s=5, alpha=0.035, rasterized=True)

    # reference
    ax.axhline(0.5, linestyle="--", linewidth=1.6)
    ax.axvspan(mp_center - mp_halfwidth, mp_center + mp_halfwidth, alpha=band_alpha)

    # key example points
    ax.scatter(
        [6.68, 6.78],
        [0.65, 0.15],
        s=120,
        edgecolor="black",
        linewidth=1.5,
        zorder=6
    )

    # annotation (tightened)
    ax.annotate(
        "ΔPI ≈ 0.50 (MP-matched)",
        xy=(6.9, 0.42),
        xytext=(7.6, 0.72),
        arrowprops=dict(arrowstyle="->", lw=1.0),
        ha="left",
        va="center"
    )

    ax.set_xlabel("Global mechanical power (J/min)")
    ax.set_ylabel("Partition Index (PI)")
    ax.set_xlim(mp.min() - 0.3, mp.max() + 0.3)
    ax.set_ylim(0.1, 0.85)

    ax.text(-0.11, 1.03, "A", transform=ax.transAxes,
            fontsize=16, fontweight="bold")

    # ===== Panel B =====
    ax = axes[1]

    valid = ~(np.isnan(pi_min) | np.isnan(pi_max))

    ax.fill_between(
        centers[valid],
        pi_min[valid],
        pi_max[valid],
        alpha=0.25
    )

    ax.plot(centers[valid], pi_min[valid], linewidth=1.0)
    ax.plot(centers[valid], pi_max[valid], linewidth=1.0)

    ax.axhline(0.5, linestyle="--", linewidth=1.6)
    ax.axvspan(mp_center - mp_halfwidth, mp_center + mp_halfwidth, alpha=band_alpha)

    ax.set_xlabel("Global mechanical power (J/min)")
    ax.set_ylabel("Partition Index (PI)")
    ax.set_xlim(mp.min() - 0.3, mp.max() + 0.3)
    ax.set_ylim(0.1, 0.85)

    ax.text(-0.10, 1.03, "B", transform=ax.transAxes,
            fontsize=16, fontweight="bold")

    # ===== Save =====
    plt.tight_layout()
    plt.savefig("Figure_4.png", dpi=600, bbox_inches="tight")
    plt.savefig("Figure_4.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
