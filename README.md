# 🌍 Climate PINN Spatial — Physics-Informed Neural Network for Climate Modeling

> Identifying climate parameters from NASA observations using a spatiotemporal Physics-Informed Neural Network.

---

## Overview

This project applies **Physics-Informed Neural Networks (PINNs)** to a real-world climate science problem: solving the **North-Sellers 1D Energy Balance Model (EBM)** — a spatiotemporal PDE — directly from NASA satellite observations.

Instead of a purely data-driven approach, the neural network is constrained to simultaneously:
- **Fit** real NASA GISS zonal temperature data (1880–2025)
- **Satisfy** a partial differential equation governing meridional heat transport
- **Identify** the climate feedback parameter λ (an inverse problem)

> A purely data-driven baseline (LSTM) was evaluated on the same dataset and achieved higher R² but failed to identify physical parameters or respect boundary conditions — motivating the physics-informed approach.

---

## The Physical Model

The 1D spatial EBM describes how temperature anomalies evolve across latitudes over time:

$$C \frac{\partial T'}{\partial t} = \kappa \frac{\partial^2 T'}{\partial x^2} - \lambda T' + F(t)$$

| Symbol | Meaning | Value / Status |
|--------|---------|----------------|
| $T'(x,t)$ | Temperature anomaly at latitude $x$, time $t$ | **network output** |
| $x = \sin(\phi)$ | Normalized latitude ∈ [-1, 1] | input |
| $C$ | Ocean-atmosphere heat capacity | fixed: **10 W·yr/m²/°C** |
| $\kappa$ | Meridional thermal diffusivity | fixed: **0.65 W/m²/°C** (North 1981) |
| $\lambda$ | Climate feedback parameter | **learned** |
| $F(t)$ | Anthropogenic forcing (CO₂ + aerosols + volcanoes) | computed |

**Boundary conditions** (no heat flux at poles):
$$\frac{\partial T'}{\partial x}\bigg|_{x=\pm 1} = 0$$

---

## Data

- **NASA GISS Surface Temperature Analysis (GISTEMP v4)** — zonal annual means
- 8 latitude bands from 90°S to 90°N
- 1880–2025 (146 years)
- **1168 spatiotemporal observations** (no registration required)
- Source: https://data.giss.nasa.gov/gistemp/

---

## Architecture

```
Input  : [x_norm, t_norm]  →  2 neurons
         Linear(2 → 64) + Tanh
         Linear(64 → 64) + Tanh
         Linear(64 → 64) + Tanh
         Linear(64 → 64) + Tanh
         Linear(64 → 1)
Output : T'_norm(x, t)     →  1 neuron

Learnable physical parameter : λ  (log-parameterized, constrained to [0.1, 3.0])
Fixed physical parameter     : κ = 0.65 W/m²/°C (North, 1981)
```

**Total trainable parameters:** 12,738

---

## Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \beta_1 \cdot \mathcal{L}_{\text{physics}} + \beta_2 \cdot \mathcal{L}_{\text{boundary}}$$

| Term | Description |
|------|-------------|
| $\mathcal{L}_{\text{data}}$ | MSE between predictions and NASA observations |
| $\mathcal{L}_{\text{physics}}$ | PDE residual at 15,000 collocation points (with Jacobian correction) |
| $\mathcal{L}_{\text{boundary}}$ | Neumann condition at poles ($\partial T'/\partial x = 0$) |

The physics residual is computed in **physical units** (W/m²) using the chain rule to convert normalized-space derivatives:

$$\frac{\partial T}{\partial t} = \frac{T_\sigma}{\Delta t} \cdot \frac{\partial u}{\partial \tau}, \qquad \frac{\partial^2 T}{\partial x^2} = \frac{T_\sigma}{\Delta x^2} \cdot \frac{\partial^2 u}{\partial \xi^2}$$

---

## Training

Two-phase training strategy with **Adam + Cosine Annealing LR** (1e-3 → 1e-5):

| Phase | Epochs | β₁ (physics weight) | β₂ (boundary weight) | Goal |
|-------|--------|---------------------|----------------------|------|
| 1 | 3,000 | **0.01** | **0.05** | Warm up — prioritize data fit |
| 2 | 7,000 | **0.15** | **0.10** | Refine physics + identify λ |

Additional constraints:
- Gradient clipping: `max_norm = 1.0`
- λ log-parameterized and clamped to physical range **[0.1, 3.0] W/m²/°C**
- κ registered as a fixed buffer (not updated by optimizer)

---

## Results

| Metric | Value | Reference |
|--------|-------|-----------|
| R² (data fit) | **0.6702** | — |
| MAE | **0.2325 °C** | — |
| RMSE | **0.3450 °C** | — |
| Max error | **2.1055 °C** | — |
| PDE residual (mean) | **0.2903 W/m²** | ideal: 0 |
| κ (fixed) | **0.6500 W/m²/°C** | North (1981): [0.55, 0.75] ✓ |
| λ identified | **0.9906 W/m²/°C** | IPCC AR6: [0.8, 1.2] ✓ |
| ECS = 3.7/λ | **3.74 °C** | IPCC AR6: [2.5, 4.0] ✓ |

**Key finding:** the climate feedback parameter λ is successfully identified from 145 years of observational data, yielding an Equilibrium Climate Sensitivity of **3.74°C per CO₂ doubling**, consistent with IPCC AR6 estimates.

**Note on R²:** the moderate R² (0.67) reflects a fundamental limitation of the EBM — not a training failure. The EBM is a deliberate simplification that cannot represent short-term variability (ENSO, volcanic eruptions, measurement noise). The 200× reduction in PDE residual (56.4 → 0.29 W/m²) compared to a data-only baseline confirms that the physics constraint is properly enforced.

**Note on κ:** meridional diffusivity is not identifiable from spatially uniform forcing alone — a known limitation in climate EBM literature (North et al., 1981). κ is fixed to its literature value.

---

## Installation

```bash
git clone https://github.com/<your-username>/climate-pinn-spatial
cd climate-pinn-spatial
pip install torch numpy pandas matplotlib scipy scikit-learn
jupyter notebook climate_pinn_spatial.ipynb
```

No API key or account needed — data is loaded automatically from NASA's public server.

---

## Project Structure

```
climate-pinn-spatial/
│
├── climate_pinn_spatial.ipynb   # Main notebook (11 sections)
├── README.md
│
└── outputs/
    ├── nasa_zonal_data.png       # Data exploration
    ├── forcing_2d.png            # Q(x,t) forcing grid
    ├── training_curves.png       # Loss + λ convergence
    ├── pinn_spatial_fit.png      # PINN vs NASA heatmaps
    ├── physics_residual.png      # PDE residual verification
    └── polar_amplification.png   # Arctic vs equatorial warming
```

---

## What Makes This Different from a Standard LSTM/DNN

| Capability | LSTM/DNN | **This project (PDE PINN)** |
|---|:---:|:---:|
| Predict at arbitrary latitude | ✗ | **✓** |
| Identify λ (climate feedback) | ✗ | **✓** |
| Respect physical boundary conditions | ✗ | **✓** |
| Spatiotemporal PDE constraint | ✗ | **✓** |
| Needs large labeled datasets | ✓ | **✗** |

---

## References

- North, G. R. et al. (1981). Energy Balance Climate Models. *Reviews of Geophysics*, 19(1), 91–121.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-Informed Neural Networks. *Journal of Computational Physics*, 378, 686–707.
- IPCC AR6 WG1 (2021). *Climate Change 2021: The Physical Science Basis*.
- NASA GISS (2024). GISS Surface Temperature Analysis (GISTEMP v4). https://data.giss.nasa.gov/gistemp/

---

## Course Context

Project for the **Deep Learning** course — demonstrating physics-informed machine learning on real scientific data.
