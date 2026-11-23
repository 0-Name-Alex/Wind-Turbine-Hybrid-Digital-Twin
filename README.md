# 🌪️ Wind Turbine Hybrid Digital Twin

<div align="center">

**A Physics-Informed Machine Learning Digital Twin & Interactive 3D Monitoring Dashboard for Industrial Wind Turbines.**  
Combines data-driven AI with first-principles thermodynamic & aerodynamic equations for early fault detection.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MATLAB](https://img.shields.io/badge/Physics%20Model-MATLAB%2FSimulink-0076A8?logo=mathworks&logoColor=white)](https://www.mathworks.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Autoencoders%20%26%20GBDT-orange)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo%20(CARE)-blue)](https://zenodo.org/records/14958989)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 🎯 Overview

Modern wind turbines operate in severe, turbulent environments where component failure causes immense downtime and repair expense. This project builds a **Hybrid Digital Twin** of real-world industrial turbines (*Wind Farm A from the CARE to Compare benchmark dataset*).

Unlike pure black-box AI or static engineering formulas, this system fuses **two independent modeling paradigms**:

```
                       ┌──────────────────────────────────────────┐
                       │           Real-time SCADA Telemetry      │
                       │   (Wind speed, Ambient temp, Active power)│
                       └────────────────────┬─────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
     ┌─────────────────────────────┐                 ┌─────────────────────────────┐
     │   🧠 Machine Learning Twin  │                 │    ⚙️ Physics-Based Twin    │
     │  - Power Curve Modeling     │                 │  - Thermal Equilibrium      │
     │  - Autoencoder Anomaly Det. │                 │  - Aerodynamic Heat Flow    │
     │  - 10m–3h Power Forecasting │                 │  - Theoretical Component    │
     │  - Degradation Health Trend │                 │    Temperature Limits       │
     └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │       ⚖️ Hybrid Comparator Engine         │
                       │   Cross-validates predictions vs physics  │
                       │   Generates plain-language fault diagnoses│
                       └────────────────────┬─────────────────────┘
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │  🖥️ Interactive 3D Streamlit Dashboard   │
                       └──────────────────────────────────────────┘
```

---

## ✨ Key Capabilities

| Module | Engine | Description |
|---|---|---|
| 🕹️ **Interactive 3D Twin** | `twin3d.py` | Full 3D turbine model with real-time color-coded component health states (Hub, Gearbox, Generator, Transformer, Pitch bearings) |
| ⚡ **Power Curve Modeling** | `power_model.py` | Physics boundary envelope combined with Gradient Boosted Decision Trees to detect aerodynamic efficiency loss |
| 🔍 **Fault & Anomaly Detection** | `fault_detection.py` | Deep autoencoders and normal-behavior models to spot uncharacteristic vibration and thermal spikes weeks before alarms trigger |
| 🌡️ **Physics Twin** | `physics_twin.py` | First-principles differential thermal equations (also available in MATLAB/Simulink under `matlab/`) |
| 🔮 **Power Forecasting** | `forecast_model.py` | Multi-horizon forecast (10 min to 3 h ahead) for grid scheduling and dispatch optimization |
| 📈 **Health & RUL Trend** | `health_trend.py` | Continuous health indexing and estimated time-to-maintenance alarm |

---

## 🚀 Quick Start

### 1️⃣ Prerequisites & Environment
Clone the repository and install dependencies:

```bash
git clone https://github.com/0-Name-Alex/Wind-Turbine-Hybrid-Digital-Twin.git
cd Wind-Turbine-Hybrid-Digital-Twin

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2️⃣ Obtain Dataset
This project is evaluated on the open-access **CARE to Compare** benchmark dataset:
1. Download from Zenodo: [**CARE to Compare Dataset (Zenodo)**](https://zenodo.org/records/14958989)
2. Extract the dataset into the repository directory under the folder name `CARE_To_Compare/`.

*(Note: Synthetic test turbines are also provided in `synthetic_turbine.py` if running offline without the full dataset).*

### 3️⃣ Launch the Digital Twin Dashboard

On Windows, simply double-click:
```powershell
START_DASHBOARD.bat
```
Or run directly from the command line:
```bash
streamlit run digital_twin_app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 📁 Repository Structure

```
Wind-Turbine-Hybrid-Digital-Twin/
├── 🌐 digital_twin_app.py      # Main Streamlit interactive dashboard
├── 🕹️ twin3d.py                # 3D interactive turbine visualizer
├── ⚙️ physics_twin.py          # Python implementation of thermodynamic physics equations
├── ⚡ power_model.py           # Aerodynamic & electrical power model
├── 🔍 fault_detection.py       # Autoencoder anomaly & fault classifier
├── 🔮 forecast_model.py        # 10 min – 3 h wind power forecaster
├── 📈 health_trend.py          # Component degradation & time-to-alarm tracker
├── 🧪 synthetic_turbine.py     # Synthetic data generator with injected fault modes
├── 📊 export_for_matlab.py     # Telemetry exporter for MATLAB/Simulink
├── 📓 RET_Digital_Twin.ipynb   # End-to-end Jupyter notebook walkthrough
├── 📁 matlab/                  # MATLAB / Simulink physics twin & block diagrams
├── 📁 outputs_*/               # Pre-computed model weights & evaluation outputs
├── 📁 results/                 # Publication-ready plots, curves & diagrams
├── 📜 requirements.txt         # Python package dependencies
└── 📜 environment.yml          # Conda environment definition
```

---

## 🔬 Physics & Mathematics Reference

The physics twin calculates expected steady-state and transient temperatures:
$$T_{\text{component}} = T_{\text{ambient}} + R_{\text{thermal}} \cdot P_{\text{loss}}(P_{\text{active}}) + \tau \cdot \frac{dT}{dt}$$

Where:
- $R_{\text{thermal}}$ is calibrated per turbine across healthy historical operation
- $P_{\text{loss}}$ accounts for copper, iron, and mechanical friction losses
- Divergence between measured $T_{\text{actual}}$ and theoretical $T_{\text{component}}$ flags cooling degradation or abnormal friction.

---

## 📄 License & Attribution

Distributed under the MIT License.  
Dataset provided by CARE to Compare under open scientific data licensing.
