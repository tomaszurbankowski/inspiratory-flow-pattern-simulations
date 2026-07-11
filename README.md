# inspiratory-flow-pattern-simulations

This repository contains the simulation code, parameter-sweep outputs, numerical-verification results, and figure-generation scripts associated with the manuscript:

**“Effects of Inspiratory Waveform, Inspiratory Time, and Pause on Compartmental Energy Routing in a Heterogeneous Two-Compartment Respiratory Model.”**

## Overview

The repository supports reproducibility of a deterministic in-silico analysis of how inspiratory flow waveform, inspiratory time, and end-inspiratory pause affect model-derived compartmental airway-level energy routing in a mechanically heterogeneous two-compartment respiratory model.

The two compartments are abstract lumped model branches rather than anatomical lung regions. The Partition Index (PI) and Energy Inequality Index (EII) should therefore be interpreted as model-derived descriptors of relative airway-level energy routing, not as direct measures of local tissue stress, strain, energy dissipation, or regional lung injury risk.

## Repository contents

### Core simulation and parameter sweep

- `mp_partitioning_v3_pattern.py`  
  Core two-compartment parallel linear resistance–compliance model under volume-controlled ventilation, with configurable inspiratory waveform and optional end-inspiratory pause.

- `run_sweep_v3_pattern.py`  
  Parameter-sweep script used to generate the main simulation dataset across waveform, inspiratory time, pause fraction, and mechanical heterogeneity conditions.

- `sweep_PI_MP_patterns.csv`  
  Main parameter-sweep output dataset containing PI, EII, compartment-specific energies, global mechanical power, and numerical-consistency metrics.

- `sweep_summary_patterns.txt`  
  Summary statistics for the main parameter sweep.

### Numerical verification

- `run_timestep_convergence.py`  
  Time-step convergence and numerical-consistency analysis. The script evaluates seven representative mechanical configurations across all 18 waveform–inspiratory-time–pause combinations and optionally repeats the sensitivity analysis over the complete 61 × 61 heterogeneity grid.

- `outputs/timestep_convergence_baseline_values.csv`  
  PI, EII, and global mechanical-power values obtained at successive time steps for a standard temporal scenario.

- `outputs/timestep_convergence_representative_detailed.csv`  
  Detailed results for all representative mechanical configurations, temporal scenarios, and tested time steps.

- `outputs/timestep_convergence_representative_summary.csv`  
  Configuration-specific maximum PI, EII, and mechanical-power differences. This file provides the numerical basis for Supplementary Table S1.

- `outputs/timestep_convergence_successive_halving.csv`  
  Maximum differences observed after each successive halving of the time step.

- `outputs/timestep_convergence_full_grid_by_scenario.csv`  
  Time-step sensitivity results for each of the 18 temporal-pattern scenarios over the complete 61 × 61 mechanical-heterogeneity grid.

- `outputs/timestep_convergence_full_grid_global_maxima.csv`  
  Global maximum PI, EII, and mechanical-power differences identified in the full-grid analysis.

- `outputs/timestep_convergence_summary.txt`  
  Text summary of the representative-case and full-grid numerical-verification results.

### Figure generation

- `generate_figure1.py`  
  Script used to generate Figure 1.

- `generate_figure2.py`  
  Script used to generate Figure 2.

- `generate_figure3.py`  
  Script used to generate Figure 3.

- `generate_figure4.py`  
  Script used to generate Figure 4.

## Model summary

The respiratory system is represented by two parallel linear resistance–compliance compartments connected to a common airway opening. Each compartment has its own resistance and compliance.

Under prescribed volume-controlled inspiratory flow, the model calculates:

- airway-opening pressure;
- dynamically partitioned compartmental flows;
- compartmental volume changes;
- branch-specific contributions to global inspiratory airway-opening work above PEEP;
- the Partition Index;
- the Energy Inequality Index;
- global inspiratory mechanical power;
- flow-balance and common-pressure consistency residuals.

During an end-inspiratory pause, total airway-opening flow is zero. Equal-and-opposite compartmental flows may nevertheless occur because of differences in compartmental resistance–compliance properties. This represents internal redistribution within the lumped model rather than additional work supplied at the airway opening.

## Default sweep settings

Unless modified by the user, the main parameter sweep uses:

- tidal volume: `VT = 0.50 L`;
- respiratory rate: `RR = 20 breaths/min`;
- PEEP: `5 cmH2O`;
- fixed time step: `dt = 0.001 s`;
- waveforms: `square`, `decelerating`, and `sinusoidal`;
- inspiratory time (`Ti`): `0.6`, `1.0`, and `1.5 s`;
- end-inspiratory pause fraction: `0.0` and `0.2`;
- baseline compartment 1 compliance: `C1 = 0.04 L/cmH2O`;
- baseline compartment 1 resistance: `R1 = 8.0 cmH2O·s/L`;
- heterogeneity range: `C2/C1` and `R2/R1` from `0.25` to `4.0`;
- grid size: `61 × 61` mechanical configurations per temporal scenario.

The combination of three waveforms, three inspiratory times, two pause conditions, and 3,721 mechanical configurations produces:

```text
3 × 3 × 2 × 61 × 61 = 66,978 simulations
```

## Numerical implementation

Airway-opening pressure is obtained directly at each time step from the parallel-flow constraint. Compartmental flows are calculated from the linear resistance–compliance equations, and compartmental volumes are advanced using a first-order explicit Euler update.

Inspiratory work and energy integrals are evaluated using the composite trapezoidal rule. Calculations use double-precision floating-point arithmetic.

The model does not use an adaptive time step or an iterative nonlinear solver. Therefore, no iterative solver tolerance applies.

The following algebraic consistency metrics are monitored:

- maximum inspiratory flow-balance residual:

```text
|F1 + F2 − Ftot|
```

- maximum discrepancy between the common airway-opening pressure and the pressure reconstructed independently from each compartment.

The prespecified numerical-consistency thresholds are:

- flow-balance residual below `1 × 10^-12 L/s`;
- common-pressure inconsistency below `1 × 10^-12 cmH2O`.

## Time-step convergence study

The representative-case convergence study uses the following time steps:

```text
4, 2, 1, 0.5, 0.25, and 0.125 ms
```

The seven representative mechanical configurations are:

1. homogeneous: `C2/C1 = 1.0`, `R2/R1 = 1.0`;
2. compliance-dominant, lower C2: `C2/C1 = 0.25`, `R2/R1 = 1.0`;
3. compliance-dominant, higher C2: `C2/C1 = 4.0`, `R2/R1 = 1.0`;
4. resistance-dominant, lower R2: `C2/C1 = 1.0`, `R2/R1 = 0.25`;
5. resistance-dominant, higher R2: `C2/C1 = 1.0`, `R2/R1 = 4.0`;
6. mixed heterogeneity, lower C2 and R2: `C2/C1 = 0.25`, `R2/R1 = 0.25`;
7. mixed heterogeneity, higher C2 and R2: `C2/C1 = 4.0`, `R2/R1 = 4.0`.

Each configuration is evaluated across all 18 temporal-pattern combinations used in the main sweep.

Convergence at the selected 1-ms time step was considered adequate when halving the time step from 1 to 0.5 ms changed both PI and EII by less than `1 × 10^-3` in absolute terms. The 0.125-ms calculation was used as a finer-step reference.

The complete 61 × 61 grid was additionally re-evaluated for all 18 temporal scenarios at time steps of 1, 0.5, and 0.125 ms.

## Numerical-verification results

Across the complete set of 66,978 simulations, reducing the time step from 1 to 0.5 ms produced maximum differences of:

- maximum absolute PI difference: `4.39 × 10^-4`;
- maximum absolute EII difference: `8.78 × 10^-4`;
- maximum relative global mechanical-power difference: `0.265%`.

Relative to the 0.125-ms finer-step reference, the maximum differences at a time step of 1 ms were:

- maximum absolute PI difference: `7.66 × 10^-4`;
- maximum absolute EII difference: `1.53 × 10^-3`;
- maximum relative global mechanical-power difference: `0.464%`.

Successive halving of the time step reduced the maximum PI and EII differences approximately twofold, consistent with the first-order explicit Euler volume update.

In the main parameter sweep, the maximum numerical-consistency residuals were:

- flow-balance residual: `2.998 × 10^-15 L/s`;
- common-pressure inconsistency: `3.553 × 10^-15 cmH2O`.

Both values were substantially below the prespecified thresholds.

## Main sweep results

The main dataset contains 66,978 simulations. The resulting ranges are:

- PI: `0.130493` to `0.801572`;
- EII: `0.000000` to `0.739014`;
- global mechanical power: `1.745860` to `13.549944 J/min`;
- maximum PI difference among conditions matched within ±5% global mechanical power: `0.501032`.

## Requirements

Recommended environment:

- Python 3.10 or newer;
- NumPy 2.0 or newer;
- pandas 2.0 or newer;
- matplotlib.

Install the required packages using:

```bash
pip install "numpy>=2.0" "pandas>=2.0" matplotlib
```

## Running the main parameter sweep

From the repository root, run:

```bash
python run_sweep_v3_pattern.py
```

This generates:

```text
sweep_PI_MP_patterns.csv
sweep_summary_patterns.txt
```

Alternative output paths and simulation settings can be specified using command-line arguments. Available options can be displayed using:

```bash
python run_sweep_v3_pattern.py --help
```

## Running the time-step convergence study

To reproduce both the representative-case analysis and the complete-grid sensitivity analysis, run:

```bash
python run_timestep_convergence.py --out-dir outputs
```

To run only the representative-case analysis and omit the computationally larger full-grid sensitivity check, use:

```bash
python run_timestep_convergence.py --out-dir outputs --skip-full-grid
```

The complete-grid analysis evaluates the full set of 66,978 conditions at each of the tested resolutions and may require several minutes, depending on the computer.

## Generating the figures

After generating or downloading the main parameter-sweep dataset, run the relevant figure-generation scripts from the repository root:

```bash
python generate_figure1.py
python generate_figure2.py
python generate_figure3.py
python generate_figure4.py
```

## Reproducibility

All scripts use deterministic calculations and do not require random-number generation. Results should therefore be reproducible across compatible Python and package versions, subject to negligible platform-specific floating-point differences.

## License

See the `LICENSE` file for licensing information.
