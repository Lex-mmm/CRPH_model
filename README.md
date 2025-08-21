# Cardiopulmonary Model Simulator

This repository contains the open-source Python implementation of the cardiopulmonary model described in our manuscript:

**"Fundamentals of modelling and simulation of physiology: A tutorial on a cardiopulmonary model"**  
*L.M. van Loon, R. Zoodsma, M.P. Mulder, R.S.P. Warnaar, E. Koomen, T. Kappen, J. Nijman, L. Fressielo, J.W. Potters, D.W. Donker, E. Oppersma (2025)*  
**Target journal:** Advances in Simulation

The tutorial and this repository together provide an accessible, stepwise introduction to physiological modelling for students, educators, and researchers.

## About the Manuscript

The manuscript presents a **five-step modelling workflow**:

1. **Model requirements** – defining scope and signals of interest
2. **Conceptual model** – structured representation of cardiovascular and respiratory compartments  
3. **Mathematical model** – formulation of governing equations (Ohm's law for flow, compliance/elastance for pressures)
4. **Software implementation and code verification** – discretisation and simulation in Python
5. **Model validation** – comparison to expected physiological values

This repository provides the complete code implementation used to generate all figures in the tutorial.

The manuscript presents a five-step modelling workflow:

Model requirements – defining scope and signals of interest

Conceptual model – structured representation of cardiovascular and respiratory compartments

Mathematical model – formulation of governing equations (Ohm’s law for flow, compliance/elastance for pressures)

Software implementation and code verification – discretisation and simulation in Python

Model validation – comparison to expected physiological values

This repository provides the full code implementation used to generate all figures in the tutorial. It allows readers to reproduce results, modify parameters, and extend the model.

## Features

**Cardiovascular System:**
- 10-compartment pulsatile heart–vascular model with time-varying elastances
- Pressure-volume loops for cardiac chambers
- Valve dynamics with backflow prevention
- Realistic hemodynamics simulation

**Respiratory System:**
- 4-compartment lung model (larynx → trachea → bronchi → alveoli)
- Airflow calculations based on pressure gradients
- Compliance-based volume dynamics
- Respiratory muscle pressure driver

**Cardio-Respiratory Coupling:**
- Intrathoracic pressure effects on cardiovascular pressures
- Physiologically realistic parameter values
- Synchronized cardiac and respiratory cycles

**Visualization:**
- Left & right heart pressure–volume loops
- Arterial, pulmonary, and venous pressures
- Respiratory flow and lung PV loop
- Driver signals (cardiac elastances, respiratory muscle pressure)
- Publication-ready figures matching the manuscript

## Requirements

```bash
numpy
matplotlib
```

Install via pip:
```bash
pip install numpy matplotlib
```

## Usage

Clone and run:

```bash
git clone https://github.com/Lex-mmm/CRPH_model.git
cd CRPH_model
python CPM_cardiorespi.py
```

This will:
- Run a 15-second baseline simulation
- Display tutorial plots interactively  
- Save all figures for comparison with the manuscript

## Output

The following figures correspond to those in the tutorial:

- `cardiorespiratory_main_plots.png` – overview (cardiac & respiratory PV loops, pressures, flow)
- `cardiac_pv_loops.png` – left & right ventricular PV loops
- `cardiovascular_pressures.png` – systemic, pulmonary, venous pressures  
- `respiratory_pv_loop.png` – lung pressure-volume relationship
- `respiratory_flow.png` – airflow waveform
- `driver_signals.png` – elastances & respiratory muscle pressure
- `cardiac_elastances.png` – cardiac elastance time series
- `respiratory_muscle_pressure.png` – respiratory muscle pressure


## License

MIT License - See [LICENSE](LICENSE) for details.