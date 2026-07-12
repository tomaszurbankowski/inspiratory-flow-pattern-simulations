#!/usr/bin/env python3
"""
Mechanical-power matching sensitivity analysis for the two-compartment sweep.

The script evaluates unordered pairs of distinct simulations using the symmetric
relative difference in global mechanical power

    delta_MP_rel(a, b) = 2 * abs(MP_a - MP_b) / (MP_a + MP_b)

and summarizes absolute between-pair differences in PI and EII for user-defined
matching tolerances. Each unordered pair is counted once.

By default, all simulation pairs are eligible, matching the analysis reported in
the manuscript. An optional mode excludes pairs generated with the same temporal
scenario (same waveform, Ti, and pause fraction).

Quantiles are accumulated with a configurable fixed histogram resolution. The
default resolution of 1e-6 is substantially finer than the three-decimal reporting
precision used in the manuscript. Pair counts and maxima are exact; quantiles are
resolved to the selected histogram precision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"MPtotal_Jmin", "PI", "EII"}
TEMPORAL_COLUMNS = {"waveform", "Ti_s", "pause_fraction"}
DEFAULT_TOLERANCES = (0.01, 0.025, 0.05, 0.10)
DEFAULT_QUANTILES = (0.25, 0.50, 0.75, 0.95)


@dataclass(frozen=True)
class AnalysisResult:
    mode: str
    tolerance: float
    pairs: int
    pi_q25: float
    pi_median: float
    pi_q75: float
    pi_p95: float
    pi_max: float
    eii_q25: float
    eii_median: float
    eii_q75: float
    eii_p95: float
    eii_max: float


def _validate_inputs(df: pd.DataFrame, modes: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    if "different_temporal_scenario" in modes:
        missing_temporal = TEMPORAL_COLUMNS.difference(df.columns)
        if missing_temporal:
            raise ValueError(
                "Mode 'different_temporal_scenario' requires columns: "
                f"{sorted(TEMPORAL_COLUMNS)}; missing: {sorted(missing_temporal)}"
            )

    for column in REQUIRED_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Column {column!r} contains non-finite or non-numeric values")
    if np.any(df["MPtotal_Jmin"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("All global mechanical-power values must be positive")


def _scenario_codes(df: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(
        df[["waveform", "Ti_s", "pause_fraction"]],
        names=["waveform", "Ti_s", "pause_fraction"],
    )
    codes, _ = pd.factorize(keys, sort=True)
    return codes.astype(np.int32, copy=False)


def _histogram_quantile(hist: np.ndarray, q: float, resolution: float) -> float:
    """Return a linearly interpolated quantile from a histogram."""
    n = int(hist.sum(dtype=np.int64))
    if n == 0:
        return float("nan")
    if not 0.0 <= q <= 1.0:
        raise ValueError("Quantile must lie in [0, 1]")

    rank = (n - 1) * q
    lo_rank = int(np.floor(rank))
    hi_rank = int(np.ceil(rank))
    weight = rank - lo_rank

    csum = np.cumsum(hist, dtype=np.int64)
    lo_bin = int(np.searchsorted(csum, lo_rank + 1, side="left"))
    hi_bin = int(np.searchsorted(csum, hi_rank + 1, side="left"))
    lo_value = lo_bin * resolution
    hi_value = hi_bin * resolution
    return float((1.0 - weight) * lo_value + weight * hi_value)


def _flush_histogram_buffer(
    bucket_parts: list[np.ndarray],
    pi_bin_parts: list[np.ndarray],
    eii_bin_parts: list[np.ndarray],
    hist_pi_flat: np.ndarray,
    hist_eii_flat: np.ndarray,
    n_bins: int,
) -> None:
    if not bucket_parts:
        return

    buckets = np.concatenate(bucket_parts)
    pi_bins = np.concatenate(pi_bin_parts)
    eii_bins = np.concatenate(eii_bin_parts)

    combined_pi = buckets.astype(np.int64) * n_bins + pi_bins
    combined_eii = buckets.astype(np.int64) * n_bins + eii_bins
    hist_pi_flat += np.bincount(combined_pi, minlength=hist_pi_flat.size)
    hist_eii_flat += np.bincount(combined_eii, minlength=hist_eii_flat.size)

    bucket_parts.clear()
    pi_bin_parts.clear()
    eii_bin_parts.clear()


def analyze_mode(
    df: pd.DataFrame,
    *,
    mode: str,
    tolerances: tuple[float, ...],
    resolution: float,
    buffer_pairs: int,
) -> list[AnalysisResult]:
    if mode not in {"all_pairs", "different_temporal_scenario"}:
        raise ValueError(f"Unsupported mode: {mode}")

    order = np.argsort(df["MPtotal_Jmin"].to_numpy(dtype=float), kind="mergesort")
    mp = df["MPtotal_Jmin"].to_numpy(dtype=float)[order]
    pi = df["PI"].to_numpy(dtype=float)[order]
    eii = df["EII"].to_numpy(dtype=float)[order]
    scenario = _scenario_codes(df)[order] if mode == "different_temporal_scenario" else None

    tolerances_arr = np.asarray(tolerances, dtype=float)
    if np.any(tolerances_arr <= 0.0) or np.any(tolerances_arr >= 2.0):
        raise ValueError("Each matching tolerance must satisfy 0 < tolerance < 2")
    if not np.all(np.diff(tolerances_arr) > 0.0):
        raise ValueError("Matching tolerances must be strictly increasing")

    max_tolerance = float(tolerances_arr[-1])
    max_ratio = (2.0 + max_tolerance) / (2.0 - max_tolerance)

    # PI and EII are normalized absolute differences and therefore lie in [0, 1].
    n_bins = int(np.ceil(1.0 / resolution)) + 1
    n_buckets = len(tolerances)
    hist_pi = np.zeros((n_buckets, n_bins), dtype=np.int64)
    hist_eii = np.zeros((n_buckets, n_bins), dtype=np.int64)
    hist_pi_flat = hist_pi.ravel()
    hist_eii_flat = hist_eii.ravel()
    bucket_max_pi = np.full(n_buckets, -np.inf, dtype=float)
    bucket_max_eii = np.full(n_buckets, -np.inf, dtype=float)

    bucket_parts: list[np.ndarray] = []
    pi_bin_parts: list[np.ndarray] = []
    eii_bin_parts: list[np.ndarray] = []
    buffered = 0

    n = len(mp)
    for i in range(n - 1):
        j_end = int(np.searchsorted(mp, mp[i] * max_ratio, side="right"))
        if j_end <= i + 1:
            continue

        sl = slice(i + 1, j_end)
        if scenario is not None:
            eligible = scenario[sl] != scenario[i]
            if not np.any(eligible):
                continue
            mp_j = mp[sl][eligible]
            pi_j = pi[sl][eligible]
            eii_j = eii[sl][eligible]
        else:
            mp_j = mp[sl]
            pi_j = pi[sl]
            eii_j = eii[sl]

        delta_mp_rel = 2.0 * (mp_j - mp[i]) / (mp_j + mp[i])
        # Numerical roundoff can place a boundary value infinitesimally above a tolerance.
        buckets = np.searchsorted(tolerances_arr, delta_mp_rel, side="left")
        within = buckets < n_buckets
        if not np.any(within):
            continue

        buckets = buckets[within].astype(np.int32, copy=False)
        delta_pi = np.abs(pi_j[within] - pi[i])
        delta_eii = np.abs(eii_j[within] - eii[i])

        # Exact maxima are kept separately; histogram bins provide quantiles.
        for bucket in np.unique(buckets):
            mask = buckets == bucket
            bucket_max_pi[bucket] = max(bucket_max_pi[bucket], float(delta_pi[mask].max()))
            bucket_max_eii[bucket] = max(bucket_max_eii[bucket], float(delta_eii[mask].max()))

        pi_bins = np.floor(delta_pi / resolution + 0.5).astype(np.int64)
        eii_bins = np.floor(delta_eii / resolution + 0.5).astype(np.int64)
        np.clip(pi_bins, 0, n_bins - 1, out=pi_bins)
        np.clip(eii_bins, 0, n_bins - 1, out=eii_bins)

        bucket_parts.append(buckets)
        pi_bin_parts.append(pi_bins)
        eii_bin_parts.append(eii_bins)
        buffered += len(buckets)

        if buffered >= buffer_pairs:
            _flush_histogram_buffer(
                bucket_parts,
                pi_bin_parts,
                eii_bin_parts,
                hist_pi_flat,
                hist_eii_flat,
                n_bins,
            )
            buffered = 0

    _flush_histogram_buffer(
        bucket_parts,
        pi_bin_parts,
        eii_bin_parts,
        hist_pi_flat,
        hist_eii_flat,
        n_bins,
    )

    # Buckets represent the smallest tolerance met. Cumulative sums yield each
    # requested tolerance because the tolerance windows are nested.
    cumulative_pi = np.cumsum(hist_pi, axis=0, dtype=np.int64)
    cumulative_eii = np.cumsum(hist_eii, axis=0, dtype=np.int64)
    cumulative_max_pi = np.maximum.accumulate(bucket_max_pi)
    cumulative_max_eii = np.maximum.accumulate(bucket_max_eii)

    results: list[AnalysisResult] = []
    for idx, tolerance in enumerate(tolerances):
        pi_hist = cumulative_pi[idx]
        eii_hist = cumulative_eii[idx]
        pairs = int(pi_hist.sum(dtype=np.int64))
        if pairs != int(eii_hist.sum(dtype=np.int64)):
            raise RuntimeError("PI and EII pair counts are inconsistent")

        values_pi = [_histogram_quantile(pi_hist, q, resolution) for q in DEFAULT_QUANTILES]
        values_eii = [_histogram_quantile(eii_hist, q, resolution) for q in DEFAULT_QUANTILES]
        results.append(
            AnalysisResult(
                mode=mode,
                tolerance=float(tolerance),
                pairs=pairs,
                pi_q25=values_pi[0],
                pi_median=values_pi[1],
                pi_q75=values_pi[2],
                pi_p95=values_pi[3],
                pi_max=float(cumulative_max_pi[idx]),
                eii_q25=values_eii[0],
                eii_median=values_eii[1],
                eii_q75=values_eii[2],
                eii_p95=values_eii[3],
                eii_max=float(cumulative_max_eii[idx]),
            )
        )
    return results


def results_to_dataframe(results: list[AnalysisResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results])


def write_text_summary(results: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "MECHANICAL-POWER MATCHING SENSITIVITY ANALYSIS",
        "Matching definition: delta_MP_rel = 2*|MP_a-MP_b|/(MP_a+MP_b)",
        "Each unordered pair of distinct simulations is counted once.",
        "Distributional summaries are descriptive.",
        "",
    ]
    for mode, group in results.groupby("mode", sort=False):
        lines.append(f"MODE: {mode}")
        for row in group.itertuples(index=False):
            lines.extend(
                [
                    f"Tolerance: {100 * row.tolerance:g}%",
                    f"  Matched pairs: {row.pairs:,}",
                    f"  |Delta PI| median (IQR): {row.pi_median:.6f} "
                    f"({row.pi_q25:.6f}-{row.pi_q75:.6f})",
                    f"  |Delta PI| p95 / max: {row.pi_p95:.6f} / {row.pi_max:.6f}",
                    f"  |Delta EII| median (IQR): {row.eii_median:.6f} "
                    f"({row.eii_q25:.6f}-{row.eii_q75:.6f})",
                    f"  |Delta EII| p95 / max: {row.eii_p95:.6f} / {row.eii_max:.6f}",
                ]
            )
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for global mechanical-power matching."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("sweep_PI_MP_patterns.csv"),
        help="Parameter-sweep CSV generated by run_sweep_v3_pattern.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("mp_matching_sensitivity_summary.csv"),
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("mp_matching_sensitivity_summary.txt"),
    )
    parser.add_argument(
        "--mode",
        choices=("all_pairs", "different_temporal_scenario", "both"),
        default="all_pairs",
        help=(
            "Pair eligibility. 'all_pairs' reproduces the manuscript analysis; "
            "'different_temporal_scenario' excludes pairs with the same waveform, Ti, "
            "and pause fraction."
        ),
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOLERANCES),
        help="Symmetric relative MP tolerances as fractions (default: 0.01 0.025 0.05 0.10).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1e-6,
        help="Histogram resolution for quantiles (default: 1e-6).",
    )
    parser.add_argument(
        "--buffer-pairs",
        type=int,
        default=2_000_000,
        help="Approximate number of pair records accumulated before histogram flushing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution <= 0.0 or args.resolution > 1.0:
        raise ValueError("--resolution must satisfy 0 < resolution <= 1")
    if args.buffer_pairs <= 0:
        raise ValueError("--buffer-pairs must be positive")

    tolerances = tuple(sorted(float(value) for value in args.tolerances))
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("Matching tolerances must be unique")

    modes = (
        ("all_pairs", "different_temporal_scenario")
        if args.mode == "both"
        else (args.mode,)
    )
    df = pd.read_csv(args.input_csv)
    _validate_inputs(df, modes)

    all_results: list[AnalysisResult] = []
    for mode in modes:
        print(f"Running mode: {mode}")
        all_results.extend(
            analyze_mode(
                df,
                mode=mode,
                tolerances=tolerances,
                resolution=float(args.resolution),
                buffer_pairs=int(args.buffer_pairs),
            )
        )

    result_df = results_to_dataframe(all_results)
    result_df.to_csv(args.output_csv, index=False, float_format="%.6f")
    write_text_summary(result_df, args.output_txt)
    print(result_df.to_string(index=False))
    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.output_txt}")


if __name__ == "__main__":
    main()
