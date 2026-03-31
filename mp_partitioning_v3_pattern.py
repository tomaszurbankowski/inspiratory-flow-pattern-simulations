#!/usr/bin/env python3
"""
mp_partitioning_v3_pattern.py

Two-compartment parallel linear RC model under VCV with configurable inspiratory
flow pattern and optional end-inspiratory pause.

The model computes inspiratory (above-PEEP) compartmental energy partitioning,
global mechanical power, and numerical consistency checks.

Units:
- Pressure: cmH2O
- Flow:     L/s
- Volume:   L
- Resistance: cmH2O*s/L
- Compliance: L/cmH2O
Energy conversion: 1 cmH2O*L = 0.098 J
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

CMH2O_L_TO_J = 0.098  # J per (cmH2O*L)
_ALLOWED_WAVEFORMS = {"square", "decelerating", "sinusoidal"}


@dataclass(frozen=True)
class VentSettings:
    VT: float = 0.50              # L
    Ti: float = 1.0               # s (total inspiratory phase, including pause)
    RR: float = 20.0              # /min
    PEEP: float = 5.0             # cmH2O
    dt: float = 0.001             # s
    waveform: str = "square"     # square | decelerating | sinusoidal
    pause_fraction: float = 0.0   # fraction of Ti occupied by end-inspiratory pause


@dataclass(frozen=True)
class Compartment:
    C: float  # L/cmH2O
    R: float  # cmH2O*s/L


def _build_inspiratory_flow_profile(vent: VentSettings, n: int) -> tuple[np.ndarray, int, int, int]:
    """
    Build total flow profile over one respiratory cycle.

    Returns
    -------
    Ftot : np.ndarray
        Total airway opening flow (L/s) over the cycle.
    n_flow : int
        Number of samples in active inspiratory flow phase.
    n_insp : int
        Number of samples in total inspiratory phase (flow + pause).
    n_pause : int
        Number of samples in end-inspiratory pause.
    """
    if vent.waveform not in _ALLOWED_WAVEFORMS:
        raise ValueError(f"waveform must be one of {_ALLOWED_WAVEFORMS}, got {vent.waveform!r}")
    if not (0.0 <= vent.pause_fraction < 1.0):
        raise ValueError("pause_fraction must satisfy 0 <= pause_fraction < 1")

    n_insp = max(2, min(int(np.round(vent.Ti / vent.dt)) + 1, n))
    flow_time = vent.Ti * (1.0 - vent.pause_fraction)
    if flow_time <= 0.0:
        raise ValueError("pause_fraction leaves no active inspiratory flow time")

    n_flow = max(2, min(int(np.round(flow_time / vent.dt)) + 1, n_insp))
    n_pause = max(0, n_insp - n_flow)

    Ftot = np.zeros(n, dtype=float)
    active_len = n_flow - 1
    t_active = np.arange(n_flow, dtype=float) * vent.dt
    active_duration = active_len * vent.dt
    if active_duration <= 0.0:
        raise ValueError("Active inspiratory duration must be positive")

    if vent.waveform == "square":
        f_active = np.full(n_flow, vent.VT / active_duration, dtype=float)
    elif vent.waveform == "decelerating":
        # Linear ramp from peak to zero over the active inspiratory interval.
        u = np.clip(t_active / active_duration, 0.0, 1.0)
        raw = 1.0 - u
        area = np.trapezoid(raw, dx=vent.dt)
        f_active = raw * (vent.VT / area)
    elif vent.waveform == "sinusoidal":
        # Positive half-sine over the active inspiratory interval.
        u = np.clip(t_active / active_duration, 0.0, 1.0)
        raw = np.sin(np.pi * u)
        area = np.trapezoid(raw, dx=vent.dt)
        f_active = raw * (vent.VT / area)
    else:
        raise AssertionError("Unhandled waveform")

    Ftot[:n_flow] = f_active
    if n_pause > 0:
        Ftot[n_flow:n_insp] = 0.0

    delivered_vt = float(np.trapezoid(Ftot[:n_insp], dx=vent.dt))
    if delivered_vt <= 0.0:
        raise RuntimeError("Generated inspiratory profile delivered non-positive VT")
    Ftot[:n_insp] *= vent.VT / delivered_vt

    return Ftot, n_flow, n_insp, n_pause


def simulate_breath_two_compartments(
    comp1: Compartment,
    comp2: Compartment,
    vent: VentSettings = VentSettings(),
    return_time_series: bool = False,
) -> Dict[str, Any]:
    """
    Simulate a single breathing cycle (one period = 60/RR) with:
      - Inspiration: prescribed Ftot(t) under VCV (square, decelerating, or sinusoidal)
        over the active flow phase, optionally followed by end-inspiratory pause.
      - Expiration: Paw clamped to PEEP, passive emptying.

    During end-inspiratory pause, the model enforces Ftot = 0 at the airway opening
    while maintaining a shared Paw across both branches.

    Returns a dict with scalar metrics and (optionally) time-series arrays.
    """
    if vent.dt <= 0:
        raise ValueError("dt must be > 0")
    if vent.Ti <= 0 or vent.RR <= 0:
        raise ValueError("Ti and RR must be > 0")
    if vent.Ti >= 60.0 / vent.RR:
        raise ValueError("Ti must be shorter than the respiratory period 60/RR")
    if comp1.C <= 0 or comp2.C <= 0 or comp1.R <= 0 or comp2.R <= 0:
        raise ValueError("All C and R must be > 0")

    T = 60.0 / vent.RR
    n = int(np.round(T / vent.dt)) + 1
    t = np.linspace(0.0, (n - 1) * vent.dt, n)

    Ftot, n_flow, n_insp, n_pause = _build_inspiratory_flow_profile(vent, n)

    Paw = np.full(n, vent.PEEP, dtype=float)
    V1 = np.zeros(n, dtype=float)
    V2 = np.zeros(n, dtype=float)
    F1 = np.zeros(n, dtype=float)
    F2 = np.zeros(n, dtype=float)

    flow_residual = []
    paw_consistency = []
    denom = (1.0 / comp1.R) + (1.0 / comp2.R)

    for k in range(1, n):
        if k < n_insp:
            numer = Ftot[k] + (V1[k - 1] / (comp1.C * comp1.R)) + (V2[k - 1] / (comp2.C * comp2.R))
            Paw_k = vent.PEEP + numer / denom

            F1_k = (Paw_k - vent.PEEP - V1[k - 1] / comp1.C) / comp1.R
            F2_k = (Paw_k - vent.PEEP - V2[k - 1] / comp2.C) / comp2.R

            V1[k] = V1[k - 1] + F1_k * vent.dt
            V2[k] = V2[k - 1] + F2_k * vent.dt

            Paw[k] = Paw_k
            F1[k] = F1_k
            F2[k] = F2_k

            r = (F1_k + F2_k) - Ftot[k]
            flow_residual.append(r)

            paw1 = vent.PEEP + (V1[k - 1] / comp1.C) + (comp1.R * F1_k)
            paw2 = vent.PEEP + (V2[k - 1] / comp2.C) + (comp2.R * F2_k)
            paw_consistency.append(max(abs(Paw_k - paw1), abs(Paw_k - paw2)))
        else:
            Paw_k = vent.PEEP
            F1_k = -V1[k - 1] / (comp1.C * comp1.R)
            F2_k = -V2[k - 1] / (comp2.C * comp2.R)

            V1[k] = V1[k - 1] + F1_k * vent.dt
            V2[k] = V2[k - 1] + F2_k * vent.dt

            Paw[k] = Paw_k
            F1[k] = F1_k
            F2[k] = F2_k

    insp_slice = slice(0, n_insp)
    dP = Paw[insp_slice] - vent.PEEP
    E1_cmh2oL = np.trapezoid(dP * F1[insp_slice], dx=vent.dt)
    E2_cmh2oL = np.trapezoid(dP * F2[insp_slice], dx=vent.dt)

    E1_J = E1_cmh2oL * CMH2O_L_TO_J
    E2_J = E2_cmh2oL * CMH2O_L_TO_J
    Etotal_J = E1_J + E2_J
    MPtotal_Jmin = Etotal_J * vent.RR

    PI = E2_J / Etotal_J if Etotal_J > 0 else np.nan
    EII = abs(E2_J - E1_J) / Etotal_J if Etotal_J > 0 else np.nan

    flow_residual = np.asarray(flow_residual, dtype=float)
    paw_consistency = np.asarray(paw_consistency, dtype=float)
    if flow_residual.size > 0:
        fr_max = float(np.max(np.abs(flow_residual)))
        fr_rms = float(np.sqrt(np.mean(flow_residual**2)))
        fr_relmax = float(fr_max / max(1e-15, float(np.max(np.abs(Ftot[:n_insp])))))
    else:
        fr_max, fr_rms, fr_relmax = float("nan"), float("nan"), float("nan")

    paw_max = float(np.max(paw_consistency)) if paw_consistency.size > 0 else float("nan")

    out: Dict[str, Any] = {
        "PI": float(PI),
        "EII": float(EII),
        "E1_J": float(E1_J),
        "E2_J": float(E2_J),
        "Etotal_J": float(Etotal_J),
        "MPtotal_Jmin": float(MPtotal_Jmin),
        "waveform": vent.waveform,
        "pause_fraction": float(vent.pause_fraction),
        "flow_time_s": float(vent.Ti * (1.0 - vent.pause_fraction)),
        "pause_time_s": float(vent.Ti * vent.pause_fraction),
        "n_flow": int(n_flow),
        "n_insp": int(n_insp),
        "n_pause": int(n_pause),
        "check_insp_flow_residual_max": fr_max,
        "check_insp_flow_residual_rms": fr_rms,
        "check_insp_flow_residual_relmax": fr_relmax,
        "check_insp_paw_consistency_max": paw_max,
    }

    if return_time_series:
        out.update({
            "t": t,
            "Paw": Paw,
            "Ftot": Ftot,
            "F1": F1,
            "F2": F2,
            "V1": V1,
            "V2": V2,
        })

    return out


if __name__ == "__main__":
    vent = VentSettings(VT=0.50, Ti=1.0, RR=20.0, PEEP=5.0, dt=0.001, waveform="decelerating", pause_fraction=0.20)
    c1 = Compartment(C=0.04, R=8.0)
    c2 = Compartment(C=0.08, R=4.0)
    res = simulate_breath_two_compartments(c1, c2, vent=vent, return_time_series=False)
    print("waveform:", res["waveform"])
    print("pause_fraction:", res["pause_fraction"])
    print("PI:", res["PI"])
    print("EII:", res["EII"])
    print("MPtotal (J/min):", res["MPtotal_Jmin"])
    print("max|F1+F2-Ftot| (L/s):", res["check_insp_flow_residual_max"])
    print("max Paw inconsistency (cmH2O):", res["check_insp_paw_consistency_max"])
