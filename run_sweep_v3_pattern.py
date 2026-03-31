#!/usr/bin/env python3
"""
run_sweep_v3_pattern.py

Sweep engine for the two-compartment model with configurable inspiratory waveform,
Ti, and end-inspiratory pause.
"""

from __future__ import annotations

import argparse
import itertools
import os
import numpy as np
import pandas as pd

from mp_partitioning_v3_pattern import VentSettings, Compartment, simulate_breath_two_compartments


def run_full_sweep(
    *,
    VT: float,
    RR: float,
    PEEP: float,
    Ti_values: list[float],
    waveforms: list[str],
    pause_fractions: list[float],
    dt: float,
    C1: float,
    R1: float,
    ratios: np.ndarray,
    out_csv: str = "sweep_PI_MP_patterns.csv",
    out_txt: str = "sweep_summary_patterns.txt",
) -> tuple[pd.DataFrame, str]:
    rows = []

    for waveform, Ti, pause_fraction in itertools.product(waveforms, Ti_values, pause_fractions):
        vent = VentSettings(
            VT=VT,
            Ti=float(Ti),
            RR=RR,
            PEEP=PEEP,
            dt=dt,
            waveform=str(waveform),
            pause_fraction=float(pause_fraction),
        )

        for C2r in ratios:
            for R2r in ratios:
                c1 = Compartment(C=C1, R=R1)
                c2 = Compartment(C=C1 * float(C2r), R=R1 * float(R2r))
                res = simulate_breath_two_compartments(c1, c2, vent=vent, return_time_series=False)
                rows.append(
                    {
                        "waveform": str(waveform),
                        "Ti_s": float(Ti),
                        "pause_fraction": float(pause_fraction),
                        "flow_time_s": float(res["flow_time_s"]),
                        "pause_time_s": float(res["pause_time_s"]),
                        "C2_over_C1": float(C2r),
                        "R2_over_R1": float(R2r),
                        "PI": float(res["PI"]),
                        "EII": float(res["EII"]),
                        "E1_J": float(res["E1_J"]),
                        "E2_J": float(res["E2_J"]),
                        "Etotal_J": float(res["Etotal_J"]),
                        "MPtotal_Jmin": float(res["MPtotal_Jmin"]),
                        "check_insp_flow_residual_max": float(res["check_insp_flow_residual_max"]),
                        "check_insp_flow_residual_rms": float(res["check_insp_flow_residual_rms"]),
                        "check_insp_flow_residual_relmax": float(res["check_insp_flow_residual_relmax"]),
                        "check_insp_paw_consistency_max": float(res["check_insp_paw_consistency_max"]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    mp = df["MPtotal_Jmin"].to_numpy(dtype=float)
    pi = df["PI"].to_numpy(dtype=float)

    order = np.argsort(mp)
    mp_s = mp[order]
    pi_s = pi[order]

    pairs = 0
    involved = set()
    max_dpi = 0.0

    for i in range(len(mp_s)):
        lo = mp_s[i] * 0.95
        hi = mp_s[i] * 1.05
        j0 = np.searchsorted(mp_s, lo, side="left")
        j1 = np.searchsorted(mp_s, hi, side="right")
        for j in range(max(i + 1, j0), j1):
            dpi = abs(pi_s[j] - pi_s[i])
            if dpi > 0.0:
                pairs += 1
                involved.add(int(order[i]))
                involved.add(int(order[j]))
                if dpi > max_dpi:
                    max_dpi = float(dpi)

    lines = [
        f"Total rows: {len(df)}",
        f"Waveforms: {', '.join(sorted(df['waveform'].unique()))}",
        f"Ti values (s): {sorted(df['Ti_s'].unique())}",
        f"Pause fractions: {sorted(df['pause_fraction'].unique())}",
        f"Grid points per scenario: {len(ratios)}x{len(ratios)} = {len(ratios) * len(ratios)}",
        f"Scenarios: {len(waveforms) * len(Ti_values) * len(pause_fractions)}",
        f"Matched pairs within ±5% MPtotal: {pairs}",
        f"Unique rows involved in MP-matched comparisons: {len(involved)}",
        f"Maximum ΔPI within ±5% MPtotal: {max_dpi:.6f}",
        f"PI range: {df['PI'].min():.6f} to {df['PI'].max():.6f}",
        f"EII range: {df['EII'].min():.6f} to {df['EII'].max():.6f}",
        f"MPtotal range (J/min): {df['MPtotal_Jmin'].min():.6f} to {df['MPtotal_Jmin'].max():.6f}",
        f"Check max residual (max over sweep): {df['check_insp_flow_residual_max'].max():.3e} L/s",
        f"Check max Paw inconsistency (max over sweep): {df['check_insp_paw_consistency_max'].max():.3e} cmH2O",
    ]
    summary = "\n".join(lines) + "\n"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    return df, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run waveform/Ti/pause sweep.")
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--vt", type=float, default=0.50)
    p.add_argument("--rr", type=float, default=20.0)
    p.add_argument("--peep", type=float, default=5.0)
    p.add_argument("--c1", type=float, default=0.04)
    p.add_argument("--r1", type=float, default=8.0)
    p.add_argument("--ti-values", type=float, nargs="+", default=[0.6, 1.0, 1.5])
    p.add_argument("--waveforms", nargs="+", default=["square", "decelerating", "sinusoidal"])
    p.add_argument("--pause-fractions", type=float, nargs="+", default=[0.0, 0.2])
    p.add_argument("--grid-n", type=int, default=61)
    p.add_argument("--grid-min", type=float, default=0.25)
    p.add_argument("--grid-max", type=float, default=4.0)
    p.add_argument("--out-csv", default="sweep_PI_MP_patterns.csv")
    p.add_argument("--out-txt", default="sweep_summary_patterns.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratios = np.linspace(float(args.grid_min), float(args.grid_max), int(args.grid_n))
    _, summary = run_full_sweep(
        VT=float(args.vt),
        RR=float(args.rr),
        PEEP=float(args.peep),
        Ti_values=[float(x) for x in args.ti_values],
        waveforms=[str(x) for x in args.waveforms],
        pause_fractions=[float(x) for x in args.pause_fractions],
        dt=float(args.dt),
        C1=float(args.c1),
        R1=float(args.r1),
        ratios=ratios,
        out_csv=str(args.out_csv),
        out_txt=str(args.out_txt),
    )
    print(summary.strip())
    print(f"Saved: {args.out_csv} and {args.out_txt}")


if __name__ == "__main__":
    main()
