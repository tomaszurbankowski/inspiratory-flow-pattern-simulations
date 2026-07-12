#!/usr/bin/env python3
"""
Time-step convergence and numerical-consistency study for the two-compartment
inspiratory energy-routing model.

The script has two parts:
1. A conventional time-step-halving study for representative mechanical cases,
   evaluated across all 18 waveform/Ti/pause scenarios used in the manuscript.
2. An optional full 61 x 61-grid sensitivity check using a vectorized evaluation
   of the same discrete equations. The vectorized implementation is verified
   against the primary scalar solver at selected grid points.

Expected companion file in the same directory:
    mp_partitioning_v3_pattern.py
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mp_partitioning_v3_pattern import (
    CMH2O_L_TO_J,
    Compartment,
    VentSettings,
    _build_inspiratory_flow_profile,
    simulate_breath_two_compartments,
)


# Both directions of one-parameter heterogeneity are included to avoid choosing
# only the numerically less demanding end of the explored range.
REPRESENTATIVE_CASES: dict[str, tuple[float, float]] = {
    "homogeneous": (1.0, 1.0),
    "compliance-dominant_low-C2": (0.25, 1.0),
    "compliance-dominant_high-C2": (4.0, 1.0),
    "resistance-dominant_low-R2": (1.0, 0.25),
    "resistance-dominant_high-R2": (1.0, 4.0),
    "mixed_low-C2-low-R2": (0.25, 0.25),
    "mixed_high-C2-high-R2": (4.0, 4.0),
}
WAVEFORMS = ("square", "decelerating", "sinusoidal")
TI_VALUES = (0.6, 1.0, 1.5)
PAUSE_FRACTIONS = (0.0, 0.2)
DT_VALUES = (0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125)
NOMINAL_DT = 0.001
REFINED_DT = 0.0005
REFERENCE_DT = 0.000125
PI_EII_TOLERANCE = 1e-3


def _scenario_iter() -> Iterable[tuple[str, float, float]]:
    return itertools.product(WAVEFORMS, TI_VALUES, PAUSE_FRACTIONS)


def run_representative_cases(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    C1, R1 = 0.04, 8.0

    for case_name, (c_ratio, r_ratio) in REPRESENTATIVE_CASES.items():
        comp1 = Compartment(C=C1, R=R1)
        comp2 = Compartment(C=C1 * c_ratio, R=R1 * r_ratio)
        for waveform, ti, pause_fraction, dt in itertools.product(
            WAVEFORMS, TI_VALUES, PAUSE_FRACTIONS, DT_VALUES
        ):
            vent = VentSettings(
                VT=0.50,
                Ti=float(ti),
                RR=20.0,
                PEEP=5.0,
                dt=float(dt),
                waveform=waveform,
                pause_fraction=float(pause_fraction),
            )
            res = simulate_breath_two_compartments(
                comp1, comp2, vent=vent, return_time_series=False
            )
            rows.append(
                {
                    "case": case_name,
                    "C2_over_C1": c_ratio,
                    "R2_over_R1": r_ratio,
                    "waveform": waveform,
                    "Ti_s": ti,
                    "pause_fraction": pause_fraction,
                    "dt_s": dt,
                    "PI": res["PI"],
                    "EII": res["EII"],
                    "MPtotal_Jmin": res["MPtotal_Jmin"],
                    "flow_residual_max_Ls": res["check_insp_flow_residual_max"],
                    "paw_consistency_max_cmH2O": res[
                        "check_insp_paw_consistency_max"
                    ],
                }
            )

    detailed = pd.DataFrame(rows)
    keys = ["case", "waveform", "Ti_s", "pause_fraction"]
    detailed.sort_values(
        keys + ["dt_s"],
        ascending=[True, True, True, True, False],
        inplace=True,
    )
    detailed.to_csv(out_dir / "timestep_convergence_representative_detailed.csv", index=False)

    nominal = detailed[detailed["dt_s"] == NOMINAL_DT].set_index(keys)
    refined = detailed[detailed["dt_s"] == REFINED_DT].set_index(keys)
    reference = detailed[detailed["dt_s"] == REFERENCE_DT].set_index(keys)

    summary_rows: list[dict[str, float | str | bool | int]] = []
    for case_name, (c_ratio, r_ratio) in REPRESENTATIVE_CASES.items():
        n = nominal.loc[case_name]
        h = refined.loc[case_name]
        r = reference.loc[case_name]

        dpi_half = np.abs(n["PI"] - h["PI"])
        deii_half = np.abs(n["EII"] - h["EII"])
        dmp_half_rel = np.abs(n["MPtotal_Jmin"] - h["MPtotal_Jmin"]) / np.abs(
            h["MPtotal_Jmin"]
        )
        dpi_ref = np.abs(n["PI"] - r["PI"])
        deii_ref = np.abs(n["EII"] - r["EII"])
        dmp_ref_rel = np.abs(n["MPtotal_Jmin"] - r["MPtotal_Jmin"]) / np.abs(
            r["MPtotal_Jmin"]
        )

        summary_rows.append(
            {
                "case": case_name,
                "C2_over_C1": c_ratio,
                "R2_over_R1": r_ratio,
                "temporal_scenarios": len(n),
                "max_abs_change_PI_1ms_to_0p5ms": float(dpi_half.max()),
                "max_abs_change_EII_1ms_to_0p5ms": float(deii_half.max()),
                "max_rel_change_MP_1ms_to_0p5ms": float(dmp_half_rel.max()),
                "max_abs_error_PI_1ms_vs_0p125ms": float(dpi_ref.max()),
                "max_abs_error_EII_1ms_vs_0p125ms": float(deii_ref.max()),
                "max_rel_error_MP_1ms_vs_0p125ms": float(dmp_ref_rel.max()),
                "max_flow_residual_Ls": float(n["flow_residual_max_Ls"].max()),
                "max_paw_consistency_cmH2O": float(
                    n["paw_consistency_max_cmH2O"].max()
                ),
                "PI_EII_refinement_pass": bool(
                    dpi_half.max() < PI_EII_TOLERANCE
                    and deii_half.max() < PI_EII_TOLERANCE
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "timestep_convergence_representative_summary.csv", index=False)

    # Actual values for a standard scenario, useful as a conventional convergence table.
    baseline = detailed[
        (detailed["waveform"] == "square")
        & (detailed["Ti_s"] == 1.0)
        & (detailed["pause_fraction"] == 0.0)
    ][
        [
            "case",
            "C2_over_C1",
            "R2_over_R1",
            "dt_s",
            "PI",
            "EII",
            "MPtotal_Jmin",
        ]
    ].copy()
    baseline.to_csv(out_dir / "timestep_convergence_baseline_values.csv", index=False)

    # Successive-halving maxima across all representative cases and temporal scenarios.
    halving_rows: list[dict[str, float]] = []
    for coarse_dt in DT_VALUES[:-1]:
        fine_dt = coarse_dt / 2.0
        coarse = detailed[detailed["dt_s"] == coarse_dt].set_index(keys)
        fine = detailed[detailed["dt_s"] == fine_dt].set_index(keys)
        halving_rows.append(
            {
                "coarse_dt_s": coarse_dt,
                "fine_dt_s": fine_dt,
                "max_abs_change_PI": float(np.abs(coarse["PI"] - fine["PI"]).max()),
                "max_abs_change_EII": float(
                    np.abs(coarse["EII"] - fine["EII"]).max()
                ),
                "max_rel_change_MP": float(
                    (
                        np.abs(coarse["MPtotal_Jmin"] - fine["MPtotal_Jmin"])
                        / np.abs(fine["MPtotal_Jmin"])
                    ).max()
                ),
            }
        )
    pd.DataFrame(halving_rows).to_csv(
        out_dir / "timestep_convergence_successive_halving.csv", index=False
    )

    return detailed, summary


def _vectorized_grid_result(
    *,
    c_ratio_grid: np.ndarray,
    r_ratio_grid: np.ndarray,
    waveform: str,
    ti: float,
    pause_fraction: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the same first-order discrete equations over an entire grid."""
    C1, R1 = 0.04, 8.0
    C2 = C1 * c_ratio_grid
    R2 = R1 * r_ratio_grid

    vent = VentSettings(
        VT=0.50,
        Ti=ti,
        RR=20.0,
        PEEP=5.0,
        dt=dt,
        waveform=waveform,
        pause_fraction=pause_fraction,
    )
    n_cycle = int(np.round((60.0 / vent.RR) / dt)) + 1
    ftot, _, n_insp, _ = _build_inspiratory_flow_profile(vent, n_cycle)

    v1 = np.zeros_like(C2, dtype=float)
    v2 = np.zeros_like(C2, dtype=float)
    denom = (1.0 / R1) + (1.0 / R2)
    e1 = np.zeros_like(C2, dtype=float)
    e2 = np.zeros_like(C2, dtype=float)
    p1_prev = np.zeros_like(C2, dtype=float)
    p2_prev = np.zeros_like(C2, dtype=float)

    for k in range(1, n_insp):
        dp = (ftot[k] + v1 / (C1 * R1) + v2 / (C2 * R2)) / denom
        f1 = (dp - v1 / C1) / R1
        f2 = (dp - v2 / C2) / R2
        p1 = dp * f1
        p2 = dp * f2
        e1 += 0.5 * (p1_prev + p1) * dt
        e2 += 0.5 * (p2_prev + p2) * dt
        v1 += f1 * dt
        v2 += f2 * dt
        p1_prev = p1
        p2_prev = p2

    e1 *= CMH2O_L_TO_J
    e2 *= CMH2O_L_TO_J
    etotal = e1 + e2
    pi = e2 / etotal
    eii = np.abs(e2 - e1) / etotal
    mp = etotal * vent.RR
    return pi, eii, mp


def _validate_vectorized_implementation(
    ratios: np.ndarray,
    c_grid_flat: np.ndarray,
    r_grid_flat: np.ndarray,
) -> None:
    # Corners, center, and two asymmetric interior points.
    indices = [
        0,
        len(ratios) - 1,
        len(ratios) * (len(ratios) // 2) + len(ratios) // 2,
        len(ratios) * 10 + 45,
        len(ratios) * 48 + 7,
    ]
    test_scenarios = [
        ("square", 1.0, 0.0, NOMINAL_DT),
        ("decelerating", 0.6, 0.2, NOMINAL_DT),
        ("sinusoidal", 1.5, 0.2, REFINED_DT),
    ]
    C1, R1 = 0.04, 8.0

    for waveform, ti, pause_fraction, dt in test_scenarios:
        pi_v, eii_v, mp_v = _vectorized_grid_result(
            c_ratio_grid=c_grid_flat,
            r_ratio_grid=r_grid_flat,
            waveform=waveform,
            ti=ti,
            pause_fraction=pause_fraction,
            dt=dt,
        )
        for idx in indices:
            c_ratio = float(c_grid_flat[idx])
            r_ratio = float(r_grid_flat[idx])
            scalar = simulate_breath_two_compartments(
                Compartment(C=C1, R=R1),
                Compartment(C=C1 * c_ratio, R=R1 * r_ratio),
                vent=VentSettings(
                    VT=0.50,
                    Ti=ti,
                    RR=20.0,
                    PEEP=5.0,
                    dt=dt,
                    waveform=waveform,
                    pause_fraction=pause_fraction,
                ),
                return_time_series=False,
            )
            if not (
                np.isclose(pi_v[idx], scalar["PI"], rtol=0.0, atol=2e-14)
                and np.isclose(eii_v[idx], scalar["EII"], rtol=0.0, atol=2e-14)
                and np.isclose(
                    mp_v[idx], scalar["MPtotal_Jmin"], rtol=0.0, atol=2e-12
                )
            ):
                raise RuntimeError("Vectorized full-grid implementation failed validation")


def run_full_grid_sensitivity(out_dir: Path) -> pd.DataFrame:
    ratios = np.linspace(0.25, 4.0, 61)
    c_mesh, r_mesh = np.meshgrid(ratios, ratios, indexing="ij")
    c_flat = c_mesh.ravel()
    r_flat = r_mesh.ravel()
    _validate_vectorized_implementation(ratios, c_flat, r_flat)

    rows: list[dict[str, float | str]] = []
    global_records: list[dict[str, float | str]] = []

    for waveform, ti, pause_fraction in _scenario_iter():
        values = {
            dt: _vectorized_grid_result(
                c_ratio_grid=c_flat,
                r_ratio_grid=r_flat,
                waveform=waveform,
                ti=ti,
                pause_fraction=pause_fraction,
                dt=dt,
            )
            for dt in (NOMINAL_DT, REFINED_DT, REFERENCE_DT)
        }

        for comparator_dt, label in (
            (REFINED_DT, "1ms_to_0p5ms"),
            (REFERENCE_DT, "1ms_vs_0p125ms"),
        ):
            nominal = values[NOMINAL_DT]
            comparator = values[comparator_dt]
            arrays = {
                "PI": np.abs(nominal[0] - comparator[0]),
                "EII": np.abs(nominal[1] - comparator[1]),
                "MP_rel": np.abs(nominal[2] - comparator[2])
                / np.abs(comparator[2]),
            }
            for metric, arr in arrays.items():
                idx = int(np.argmax(arr))
                global_records.append(
                    {
                        "comparison": label,
                        "metric": metric,
                        "maximum": float(arr[idx]),
                        "waveform": waveform,
                        "Ti_s": ti,
                        "pause_fraction": pause_fraction,
                        "C2_over_C1": float(c_flat[idx]),
                        "R2_over_R1": float(r_flat[idx]),
                    }
                )

        rows.append(
            {
                "waveform": waveform,
                "Ti_s": ti,
                "pause_fraction": pause_fraction,
                "max_abs_change_PI_1ms_to_0p5ms": float(
                    np.max(np.abs(values[NOMINAL_DT][0] - values[REFINED_DT][0]))
                ),
                "max_abs_change_EII_1ms_to_0p5ms": float(
                    np.max(np.abs(values[NOMINAL_DT][1] - values[REFINED_DT][1]))
                ),
                "max_rel_change_MP_1ms_to_0p5ms": float(
                    np.max(
                        np.abs(values[NOMINAL_DT][2] - values[REFINED_DT][2])
                        / np.abs(values[REFINED_DT][2])
                    )
                ),
                "max_abs_error_PI_1ms_vs_0p125ms": float(
                    np.max(np.abs(values[NOMINAL_DT][0] - values[REFERENCE_DT][0]))
                ),
                "max_abs_error_EII_1ms_vs_0p125ms": float(
                    np.max(np.abs(values[NOMINAL_DT][1] - values[REFERENCE_DT][1]))
                ),
                "max_rel_error_MP_1ms_vs_0p125ms": float(
                    np.max(
                        np.abs(values[NOMINAL_DT][2] - values[REFERENCE_DT][2])
                        / np.abs(values[REFERENCE_DT][2])
                    )
                ),
            }
        )

    scenario_summary = pd.DataFrame(rows)
    scenario_summary.to_csv(
        out_dir / "timestep_convergence_full_grid_by_scenario.csv", index=False
    )

    record_df = pd.DataFrame(global_records)
    global_summary = (
        record_df.sort_values("maximum", ascending=False)
        .groupby(["comparison", "metric"], as_index=False)
        .first()
    )
    global_summary.to_csv(
        out_dir / "timestep_convergence_full_grid_global_maxima.csv", index=False
    )
    return global_summary


def write_text_summary(
    out_dir: Path,
    representative_summary: pd.DataFrame,
    full_grid_summary: pd.DataFrame | None,
) -> None:
    max_rep = representative_summary[
        [
            "max_abs_change_PI_1ms_to_0p5ms",
            "max_abs_change_EII_1ms_to_0p5ms",
            "max_rel_change_MP_1ms_to_0p5ms",
            "max_abs_error_PI_1ms_vs_0p125ms",
            "max_abs_error_EII_1ms_vs_0p125ms",
            "max_rel_error_MP_1ms_vs_0p125ms",
            "max_flow_residual_Ls",
            "max_paw_consistency_cmH2O",
        ]
    ].max()

    lines = [
        "TIME-STEP CONVERGENCE AND NUMERICAL-CONSISTENCY STUDY",
        f"Fixed-step sequence (s): {', '.join(f'{x:g}' for x in DT_VALUES)}",
        f"Nominal time step: {NOMINAL_DT:g} s",
        f"Fine-grid reference: {REFERENCE_DT:g} s",
        f"Convergence criterion for PI and EII: maximum absolute change < {PI_EII_TOLERANCE:g}",
        f"when dt is halved from {NOMINAL_DT:g} to {REFINED_DT:g} s.",
        "",
        "REPRESENTATIVE CASES (7 mechanical cases x 18 temporal scenarios)",
        f"Max |PI(1 ms)-PI(0.5 ms)|: {max_rep['max_abs_change_PI_1ms_to_0p5ms']:.9g}",
        f"Max |EII(1 ms)-EII(0.5 ms)|: {max_rep['max_abs_change_EII_1ms_to_0p5ms']:.9g}",
        f"Max relative MP change, 1 ms to 0.5 ms: {100*max_rep['max_rel_change_MP_1ms_to_0p5ms']:.6f}%",
        f"Max |PI(1 ms)-PI(0.125 ms)|: {max_rep['max_abs_error_PI_1ms_vs_0p125ms']:.9g}",
        f"Max |EII(1 ms)-EII(0.125 ms)|: {max_rep['max_abs_error_EII_1ms_vs_0p125ms']:.9g}",
        f"Max relative MP error, 1 ms vs 0.125 ms: {100*max_rep['max_rel_error_MP_1ms_vs_0p125ms']:.6f}%",
        f"Max flow-balance residual at 1 ms: {max_rep['max_flow_residual_Ls']:.3e} L/s",
        f"Max common-pressure inconsistency at 1 ms: {max_rep['max_paw_consistency_cmH2O']:.3e} cmH2O",
        f"All representative cases passed: {bool(representative_summary['PI_EII_refinement_pass'].all())}",
    ]

    if full_grid_summary is not None:
        def _get(comp: str, metric: str) -> float:
            return float(
                full_grid_summary[
                    (full_grid_summary["comparison"] == comp)
                    & (full_grid_summary["metric"] == metric)
                ]["maximum"].iloc[0]
            )

        lines += [
            "",
            "FULL 61 x 61 GRID (66,978 simulations at each tested resolution)",
            f"Max |PI(1 ms)-PI(0.5 ms)|: {_get('1ms_to_0p5ms', 'PI'):.9g}",
            f"Max |EII(1 ms)-EII(0.5 ms)|: {_get('1ms_to_0p5ms', 'EII'):.9g}",
            f"Max relative MP change, 1 ms to 0.5 ms: {100*_get('1ms_to_0p5ms', 'MP_rel'):.6f}%",
            f"Max |PI(1 ms)-PI(0.125 ms)|: {_get('1ms_vs_0p125ms', 'PI'):.9g}",
            f"Max |EII(1 ms)-EII(0.125 ms)|: {_get('1ms_vs_0p125ms', 'EII'):.9g}",
            f"Max relative MP error, 1 ms vs 0.125 ms: {100*_get('1ms_vs_0p125ms', 'MP_rel'):.6f}%",
        ]

    (out_dir / "timestep_convergence_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run time-step convergence and numerical-consistency checks."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated convergence outputs (default: outputs).",
    )
    parser.add_argument(
        "--skip-full-grid",
        action="store_true",
        help="Run only the representative-case study.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, representative_summary = run_representative_cases(args.out_dir)
    full_grid_summary = None
    if not args.skip_full_grid:
        full_grid_summary = run_full_grid_sensitivity(args.out_dir)
    write_text_summary(args.out_dir, representative_summary, full_grid_summary)


if __name__ == "__main__":
    main()
