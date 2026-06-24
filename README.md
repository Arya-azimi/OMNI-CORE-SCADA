# 🌐 MAGNORA | OMNI-CORE SCADA System

## ⚡ Executive Summary

Magnora OMNI-CORE is an industrial-grade, high-performance SCADA system integrated with a proprietary Virtual Core AI Engine. By leveraging deep learning (PyTorch) and high-performance tabular regression (XGBoost), this platform replaces expensive physical core laboratory analysis with real-time, AI-driven porosity and lithology predictions derived directly from streaming wireline log data.

## 🚀 Key Capabilities

- Virtual Coring AI: Eliminates the need for costly physical core extraction and laboratory testing by predicting rock properties in real-time.

- Holographic Telemetry: Advanced 3D trajectory visualization featuring neon-glowing drill paths and real-time operational depth mapping.

- Low-Latency Streaming: WebSocket-powered architecture designed for high-frequency industrial data processing and instant field feedback.

- Real-time Target Detection: Intelligent visualization of target payzones (porosity reservoirs) using semi-transparent holographic planes.

- Cyberpunk Dashboard: A sophisticated glassmorphism UI designed for rapid operational decision-making in demanding field environments.

## 🛠️ Tech Stack & Architecture

- Backend Engine: FastAPI (Async concurrency for high-throughput data management).

### AI/ML Core:

- PyTorch: Deep Feature Extractor for latent representation learning.

- XGBoost: High-speed tabular regression for precise petrophysical output.

- Data Processing: NumPy & Pandas (Efficient vectorization of telemetry data).

- Frontend: Plotly.js (WebGL Accelerated) for high-performance 3D rendering and petrophysical log visualization.

## 📊 Operational & Data Diagrams

### 1. Sensor Data Processing

- The system ingests and processes high-frequency telemetry logs, including:

- GR (Gamma Ray): Lithology classification.

- RHOB (Bulk Density): Reservoir density determination.

- NPHI (Neutron Porosity): Identification of fluid-filled pore spaces.

- DT (Sonic): Acoustic property analysis.

- RESD (Resistivity): Fluid saturation detection.

### 2. Petrophysical Logging & Crossplots

- The platform generates real-time Crossplots (NPHI vs. RHOB), enabling field engineers to instantly classify rock types (Sandstone, Limestone, or Dolomite) based on AI-filtered telemetry.

### 3. Holographic 3D Visualization

- The core visualizer provides an immersive radar-like experience:

- Well Trajectory: Dynamic neon-rendered drill paths.

- Payzone Plane: Semi-transparent target reservoirs at depth (e.g., 5800 ft).

- Drill Bit Marker: Real-time marker tracking the drill bit's spatial and depth coordinates.

## 📦 Getting Started

### 1. Prerequisites

Ensure Python 3.12+ is installed on your system.
```
git clone https://github.com/Arya-azimi/OMNI-CORE-SCADA.git
cd OMNI-CORE-SCADA
```


### 2. Environment Setup

Install the necessary high-performance compute and web libraries:
```
pip install -r requirements.txt
```


### 3. Launching the System

Run the main FastAPI server:
```
cd app
python app.py
```


Once active, navigate to http://0.0.0.0:8000 to access the OMNI-CORE dashboard.

## 🏗️ Project Structure
```
├── app/
│   └── app_fastapi.py    # Main FastAPI / WebSocket engine
├── models/
│   ├── xgboost_virtual_core.json # Trained AI prediction weights
│   └── virtual_core_scaler.pkl   # Data normalization parameters
├── data/                 # Live sensor stream buffers
└── requirements.txt      # Dependency manifest
```



## ⚙️ AI Engine Implementation

The system utilizes a DeepFeatureExtractor (PyTorch) to map noisy field-acquired wireline logs into robust feature spaces. These features are then passed through an XGBRegressor to calculate highly accurate porosity estimates. The entire pipeline is architected for maximum noise immunity in rugged field conditions.

## 🤝 Contribution & Licensing

This project is part of the Magnora Neural Stratigraphy initiative.

Developer: Arya

License: MIT

Full Dataset: https://wiki.seg.org/wiki/Teapot_dome_3D_survey

Built for the future of automated, AI-driven energy exploration.
