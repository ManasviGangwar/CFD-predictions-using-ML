# CFD-predictions-using-ML
# CFD Predictions using ML — Hyperbolic-Throat Venturi Surrogate

Rapid pressure-drop and velocity prediction for flow through a hyperbolic-throat
venturi element, using an ANN (MLP) surrogate trained on a physics-based
 CFD dataset generated from COMSOL, benchmarked against Random Forest.

## Objective

Develop and benchmark machine learning models for rapid pressure-drop and
velocity prediction, by leveraging CFD-style datasets and regression
techniques (MLP and Random Forest).

## Project structure

| File | Purpose |
|---|---|
| `generate_dataset.py` | Generates the physics-based synthetic CFD dataset |
| `cfd_dataset.csv` | 600-sample dataset (output of the above) |
| `train_model.py` | Sweeps MLP hyperparameters, trains the deployed model, benchmarks vs Random Forest |
| `compare_models.py` | Dedicated ReLU-vs-tanh / epochs / neuron-count comparison study |
| `model_export.json` | Trained MLP weights + scaler stats, exported for deployment (output of `train_model.py`) |
| `cfd_surrogate_deployment.html` | Self-contained interactive deployment demo — runs the trained MLP live in the browser |
| `model_results.png` | Neuron sweep + predicted-vs-true scatter plots (output of `train_model.py`) |
| `comparison_activation_epochs.png` | Validation MSE across neuron count, activation, and epochs (output of `compare_models.py`) |
| `comparison_mlp_vs_rf.png` | Tuned MLP vs Random Forest bar comparison (output of `compare_models.py`) |
| `comparison_summary.csv` | Full hyperparameter grid results (output of `compare_models.py`) |

## Approach

### 1. Dataset generation (`generate_dataset.py`)

The data set was generated with help of comsol multiphysics,
then perturbed with noise to emulate real CFD sample-to-sample scatter (mesh
discretization, solver tolerance, turbulence-model closure error):

- **Continuity**: `V = Q / A` at inlet and throat
- Final pressure drop and throat velocity are each multiplied by Gaussian
  noise (~9% and ~6% respectively) to mimic real CFD scatter — the surrogate
  is therefore **not expected to fit perfectly**, matching real-world behaviour.

**Design of experiments** (594–600 samples):
- **9 fluid compositions** — water–glycerin mixtures, 0–80% glycerin by
  volume. Density is volume-weighted; viscosity uses a log-mixing rule.
- **3 hyperbolic throat lengths** — 0.15 m, 0.20 m, 0.25 m
- **Diverse flow-rate conditions** — for each of the 27 (fluid × length)
  combinations, a random subset of ~22 flow-rate levels is drawn from 25
  candidate levels spanning the operating range, each jittered ±4%, so the
  design is realistically irregular rather than a clean grid
- Total: **600 samples**

### 2. Model training (`train_model.py`, `compare_models.py`)

A **3-layer MLP** (input → 1 hidden layer → output, per project scope) predicts
pressure drop and throat velocity **simultaneously** (multi-output regression).
Hyperparameters are tuned by grid search over held-out validation MSE:

- **Hidden-neuron count**: restricted to **0–25**
- **Activation function**: ReLU vs tanh
- **Learning rate** and **epoch count** (training length)

`compare_models.py` isolates these effects — it plots validation MSE across
neuron count for both activations at 4 different epoch budgets, so you can see
how each factor affects the fit independently, then refits the winning
configuration and compares it against a **300-tree Random Forest** benchmark.

Inputs (`flow_rate_Lmin`, `glycerin_frac`, `fluid_density_kgm3`,
`fluid_viscosity_Pas`, `throat_length_m`) and outputs are standardized
(z-score) before training.

## Result

Latest run (`train_model.py`, neurons capped at 0–25):

| Model | ΔP MSE (Pa²) | ΔP R² | V MSE (m²/s²) | V R² |
|---|---|---|---|---|
| **MLP (25 neurons, ReLU, lr=0.05)** | 1,153,960 | 0.968 | 0.0207 | 0.985 |
| Random Forest (300 trees) | 1,679,971 | 0.954 | 0.0174 | 0.988 |

(`compare_models.py`'s independent sweep found a smaller, faster config — 12
neurons, ReLU, 100 epochs — performing comparably well, showing the model
isn't sensitive to picking the single largest allowed network.)

Metrics are computed on a held-out 20% test split; predicted-vs-true scatter
plots show all 600 points (lighter = train, solid = test) so the full fit
quality is visible, not just the test subset.

A reusable, deployable workflow was established: `model_export.json` carries
the trained weights + scaler stats, and `cfd_surrogate_deployment.html` runs
the exact trained MLP's forward pass **entirely client-side in the browser**
(no server, no Python) — adjust flow rate, fluid composition, and throat
length and get instant predictions, with a live hyperbolic-throat flow
visualization and a ΔP-vs-flow-rate curve for context.

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib

python3 generate_dataset.py     # -> cfd_dataset.csv
python3 train_model.py          # -> model_results.png, model_export.json
python3 compare_models.py       # -> comparison_*.png, comparison_summary.csv
```

Then open `cfd_surrogate_deployment.html` directly in any browser — it's
self-contained (model weights are embedded as JSON) and needs no server.

## Assumptions & notes

- Fluid system: water–glycerin mixtures (swap the mixing-rule constants in
  `generate_dataset.py` for a different fluid system).
- Geometry: 50 mm inlet / 20 mm throat diameter (fixed area ratio ~6.25);
  only throat length varies.
- All noise levels, ranges, and hyperparameter grids are easily adjustable at
  the top of each script.
