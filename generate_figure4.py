#!/usr/bin/env python3
"""
generate_figure4.py

Submission-ready Figure 4 generated from sweep_PI_MP_patterns.csv.

Panel A
-------
Scatterplot of Partition Index (PI) versus global mechanical power (MP).
The highlighted pair is identified automatically as the unordered pair with
the largest absolute PI difference among all pairs satisfying the manuscript's
symmetric relative MP-matching criterion:

    2 * |MP_a - MP_b| / (MP_a + MP_b) <= tolerance

The default tolerance is 5%.

Panel B
-------
Descriptive minimum-to-maximum PI envelope within equal-width MP bins.
This panel is an MP-binned envelope and is not itself the 5% matching rule.

Outputs
-------
Figure_4.png
Figure_4.svg
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"MPtotal_Jmin", "PI"}


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate columns and numerical values required for Figure 4."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    for column in sorted(REQUIRED_COLUMNS):
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Column {column!r} contains non-numeric or non-finite values."
            )

    if len(df) < 2:
        raise ValueError("At least two simulations are required.")

    if np.any(df["MPtotal_Jmin"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("All global mechanical-power values must be positive.")


def find_max_delta_pi_pair(
    mp: np.ndarray,
    pi: np.ndarray,
    tolerance: float,
) -> tuple[int, int, float, float]:
    """
    Find the exact MP-matched pair with the largest absolute PI difference.

    The arrays are sorted by MP. A sliding window and monotonic deques track
    the minimum and maximum PI among eligible earlier simulations, giving an
    exact O(n log n) sorting step followed by an O(n) scan.

    Returns
    -------
    index_a, index_b
        Indices in the original input arrays.
    delta_pi
        Absolute PI difference.
    delta_mp_rel
        Symmetric relative MP difference.
    """
    if not 0.0 < tolerance < 2.0:
        raise ValueError("Matching tolerance must satisfy 0 < tolerance < 2.")

    order = np.argsort(mp, kind="mergesort")
    mp_sorted = mp[order]
    pi_sorted = pi[order]

    max_ratio = (2.0 + tolerance) / (2.0 - tolerance)

    min_pi_deque: deque[int] = deque()
    max_pi_deque: deque[int] = deque()
    left = 0

    best_delta_pi = -np.inf
    best_sorted_i = -1
    best_sorted_j = -1

    for j in range(len(mp_sorted)):
        lower_mp = mp_sorted[j] / max_ratio

        while left < j and mp_sorted[left] < lower_mp:
            if min_pi_deque and min_pi_deque[0] == left:
                min_pi_deque.popleft()
            if max_pi_deque and max_pi_deque[0] == left:
                max_pi_deque.popleft()
            left += 1

        if min_pi_deque:
            i = min_pi_deque[0]
            delta = abs(pi_sorted[j] - pi_sorted[i])
            if delta > best_delta_pi:
                best_delta_pi = float(delta)
                best_sorted_i = i
                best_sorted_j = j

        if max_pi_deque:
            i = max_pi_deque[0]
            delta = abs(pi_sorted[j] - pi_sorted[i])
            if delta > best_delta_pi:
                best_delta_pi = float(delta)
                best_sorted_i = i
                best_sorted_j = j

        while min_pi_deque and pi_sorted[min_pi_deque[-1]] >= pi_sorted[j]:
            min_pi_deque.pop()
        min_pi_deque.append(j)

        while max_pi_deque and pi_sorted[max_pi_deque[-1]] <= pi_sorted[j]:
            max_pi_deque.pop()
        max_pi_deque.append(j)

    if best_sorted_i < 0 or best_sorted_j < 0:
        raise RuntimeError(
            "No distinct simulation pair satisfied the selected MP tolerance."
        )

    original_i = int(order[best_sorted_i])
    original_j = int(order[best_sorted_j])

    mp_i = float(mp[original_i])
    mp_j = float(mp[original_j])
    delta_mp_rel = 2.0 * abs(mp_i - mp_j) / (mp_i + mp_j)

    return original_i, original_j, best_delta_pi, float(delta_mp_rel)


def calculate_binned_envelope(
    mp: np.ndarray,
    pi: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate minimum and maximum PI in equal-width MP bins."""
    if n_bins < 2:
        raise ValueError("The number of MP bins must be at least 2.")

    edges = np.linspace(float(mp.min()), float(mp.max()), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_index = np.searchsorted(edges, mp, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    pi_min = np.full(n_bins, np.nan, dtype=float)
    pi_max = np.full(n_bins, np.nan, dtype=float)

    for idx in range(n_bins):
        values = pi[bin_index == idx]
        if values.size:
            pi_min[idx] = float(values.min())
            pi_max[idx] = float(values.max())

    return centers, pi_min, pi_max


def generate_figure(
    input_csv: Path,
    output_png: Path,
    output_svg: Path,
    tolerance: float,
    n_bins: int,
) -> None:
    df = pd.read_csv(input_csv)
    validate_dataframe(df)

    mp = df["MPtotal_Jmin"].to_numpy(dtype=float)
    pi = df["PI"].to_numpy(dtype=float)

    idx_a, idx_b, delta_pi, delta_mp_rel = find_max_delta_pi_pair(
        mp=mp,
        pi=pi,
        tolerance=tolerance,
    )

    pair_mp = np.array([mp[idx_a], mp[idx_b]], dtype=float)
    pair_pi = np.array([pi[idx_a], pi[idx_b]], dtype=float)

    centers, pi_min, pi_max = calculate_binned_envelope(
        mp=mp,
        pi=pi,
        n_bins=n_bins,
    )
    valid = np.isfinite(pi_min) & np.isfinite(pi_max)

    cloud_color = "#1f77b4"
    envelope_color = "#1f77b4"
    reference_color = "0.35"
    highlight_edge = "black"
    highlight_face = "white"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    ax = axes[0]
    ax.scatter(
        mp,
        pi,
        s=5,
        color=cloud_color,
        alpha=0.035,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    ax.axhline(
        0.5,
        color=reference_color,
        linestyle="--",
        linewidth=1.4,
        zorder=2,
    )
    ax.plot(
        pair_mp,
        pair_pi,
        color=highlight_edge,
        linestyle="--",
        linewidth=1.2,
        zorder=5,
    )
    ax.scatter(
        pair_mp,
        pair_pi,
        s=105,
        facecolor=highlight_face,
        edgecolor=highlight_edge,
        linewidth=1.5,
        zorder=6,
    )

    midpoint_x = float(pair_mp.mean())
    midpoint_y = float(pair_pi.mean())
    text_x = min(float(mp.max()) - 0.2, midpoint_x + 0.10 * np.ptp(mp))
    text_y = min(0.80, midpoint_y + 0.20)

    ax.annotate(
        (
            f"|ΔPI| = {delta_pi:.3f}\n"
            f"ΔMP$_{{rel}}$ = {100.0 * delta_mp_rel:.2f}%"
        ),
        xy=(midpoint_x, midpoint_y),
        xytext=(text_x, text_y),
        arrowprops=dict(arrowstyle="->", color=highlight_edge, lw=1.0),
        ha="left",
        va="center",
        fontsize=10,
        zorder=7,
    )

    ax.set_xlabel("Global mechanical power (J/min)")
    ax.set_ylabel("Partition Index (PI)")
    ax.set_xlim(float(mp.min()) - 0.3, float(mp.max()) + 0.3)
    ax.set_ylim(0.1, 0.85)
    ax.text(
        -0.11,
        1.03,
        "A",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
    )

    ax = axes[1]
    ax.fill_between(
        centers[valid],
        pi_min[valid],
        pi_max[valid],
        color=envelope_color,
        alpha=0.22,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        centers[valid],
        pi_min[valid],
        color=envelope_color,
        linewidth=1.0,
        zorder=2,
    )
    ax.plot(
        centers[valid],
        pi_max[valid],
        color=envelope_color,
        linewidth=1.0,
        zorder=2,
    )
    ax.axhline(
        0.5,
        color=reference_color,
        linestyle="--",
        linewidth=1.4,
        zorder=3,
    )

    ax.set_xlabel("Global mechanical power (J/min)")
    ax.set_ylabel("Partition Index (PI)")
    ax.set_xlim(float(mp.min()) - 0.3, float(mp.max()) + 0.3)
    ax.set_ylim(0.1, 0.85)
    ax.text(
        -0.10,
        1.03,
        "B",
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    print(
        "Highlighted matched pair:\n"
        f"  row indices: {idx_a}, {idx_b}\n"
        f"  MP: {pair_mp[0]:.6f}, {pair_mp[1]:.6f} J/min\n"
        f"  PI: {pair_pi[0]:.6f}, {pair_pi[1]:.6f}\n"
        f"  |Delta PI|: {delta_pi:.6f}\n"
        f"  Delta MP_rel: {100.0 * delta_mp_rel:.6f}%\n"
        f"Saved: {output_png}\n"
        f"Saved: {output_svg}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 4 from the parameter-sweep CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("sweep_PI_MP_patterns.csv"),
        help="Input sweep CSV.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("Figure_4.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=Path("Figure_4.svg"),
        help="Output SVG path.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Symmetric relative MP-matching tolerance as a fraction (default: 0.05).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=119,
        help="Number of equal-width MP bins for Panel B (default: 119).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_figure(
        input_csv=args.input_csv,
        output_png=args.output_png,
        output_svg=args.output_svg,
        tolerance=float(args.tolerance),
        n_bins=int(args.bins),
    )


if __name__ == "__main__":
    main()
