# regional-energy-routing-simulations

This repository contains simulation scripts, parameter-sweep outputs, and figure-generation code for analyses of regional inspiratory energy routing in heterogeneous lung models.

## Overview

The repository supports reproducibility of the analyses and figures reported in the associated manuscript on inspiratory flow pattern, inspiratory time, end-inspiratory pause, and regional inspiratory energy partitioning in a two-compartment heterogeneous lung model.

## Repository contents

- `mp_partitioning_v3_pattern.py`  
  Core two-compartment parallel linear RC model under volume-controlled ventilation, with configurable inspiratory waveform and optional end-inspiratory pause.

- `run_sweep_v3_pattern.py`  
  Parameter-sweep script used to generate the main simulation dataset across waveform, inspiratory time, pause fraction, and mechanical heterogeneity conditions.

- `sweep_PI_MP_patterns.csv`  
  Main parameter-sweep output dataset.

- `sweep_summary_patterns.txt`  
  Summary statistics for the full sweep.

- `generate_figure1.py`  
  Script to generate Figure 1.

- `generate_figure2.py`  
  Script to generate Figure 2.

- `generate_figure3.py`  
  Script to generate Figure 3.

- `generate_figure4.py`  
  Script to generate Figure 4.

## Model summary

The model represents the respiratory system as two parallel compartments with distinct resistance and compliance values. Under prescribed inspiratory flow at the airway opening, the code computes compartment-specific inspiratory flow, cumulative inspiratory energy delivered above PEEP, the Partition Index (PI), the Energy Inequality Index (EII), and global mechanical power.

## Default sweep settings

Unless modified by the user, the sweep uses:

- Waveforms: `square`, `decelerating`, `sinusoidal`
- Inspiratory time (`Ti`): 0.6 s, 1.0 s, 1.5 s
- End-inspiratory pause fraction: 0.0, 0.2
- Baseline compartment 1 parameters: `C1 = 0.04 L/cmH2O`, `R1 = 8.0 cmH2O·s/L`
- Heterogeneity grid: `C2/C1` and `R2/R1` from 0.25 to 4.0
- Grid size: 61 × 61 points per scenario

## Main outputs

The full sweep includes 66,978 rows across 18 scenarios. In the current dataset:

- PI range: 0.130493 to 0.801572
- EII range: 0.000000 to 0.739014
- Global mechanical power range: 1.745860 to 13.549944 J/min
- Maximum ΔPI within ±5% matched global mechanical power: 0.501032

## Requirements

Recommended environment:

- Python 3.10+
- numpy
- pandas
- matplotlib

## Installation

Create an environment and install the required packages:

```bash
pip install numpy pandas matplotlib
