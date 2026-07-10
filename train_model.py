"""
Train a surrogate model for dual-output prediction (pressure drop, throat
velocity) from the physics-based CFD dataset.

Per the project spec: a single-hidden-layer MLP (3 layers total: input,
hidden, output), tuned by "synthesizing insights of hyperparameter tuning
and optimizing the hidden-layer neuron count through minimum-MSE analysis."

Hyperparameters swept (single hidden layer only):
  - hidden-layer neuron count
  - activation function (relu / tanh / logistic)
  - learning rate (init)
  - epochs (max_iter)

A Random Forest is trained alongside as a benchmark. Because the training
data carries realistic simulation noise, neither model is expected to fit
perfectly -- this mirrors real CFD-surrogate behaviour.
"""
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

RNG = 7
df = pd.read_csv("cfd_dataset.csv", comment="#")

FEATURES = ["flow_rate_Lmin", "glycerin_frac", "fluid_density_kgm3",
            "fluid_viscosity_Pas", "throat_length_m"]
TARGETS = ["pressure_drop_Pa", "throat_velocity_ms"]

X = df[FEATURES].values
y = df[TARGETS].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG)

x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_s = x_scaler.transform(X_train)
X_test_s = x_scaler.transform(X_test)
y_train_s = y_scaler.transform(y_train)
y_test_s = y_scaler.transform(y_test)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_s, y_train_s, test_size=0.2, random_state=RNG)

# ---------------------------------------------------------------------
# Hyperparameter sweep -- single hidden layer only, per project spec
# ---------------------------------------------------------------------
neuron_grid = [2, 4, 6, 8, 10, 12, 15, 18, 20, 22, 25]
activation_grid = ["relu", "tanh", "logistic"]
lr_grid = [0.001, 0.01, 0.05]
epoch_grid = [200, 500, 1000]

results = []
for n, act, lr, ep in itertools.product(neuron_grid, activation_grid, lr_grid, epoch_grid):
    mlp = MLPRegressor(hidden_layer_sizes=(n,), activation=act,
                        solver="adam", learning_rate_init=lr, max_iter=ep,
                        random_state=RNG)
    mlp.fit(X_tr, y_tr)
    pred = mlp.predict(X_val)
    mse = mean_squared_error(y_val, pred)
    results.append({"neurons": n, "activation": act, "lr": lr,
                     "epochs": ep, "val_mse": mse})

results_df = pd.DataFrame(results).sort_values("val_mse").reset_index(drop=True)
best = results_df.iloc[0]
print("Top 8 hyperparameter combinations (by validation MSE, scaled):")
print(results_df.head(8).to_string(index=False))

best_n, best_act, best_lr, best_ep = (
    int(best.neurons), best.activation, float(best.lr), int(best.epochs))
print(f"\nSelected: neurons={best_n}, activation={best_act}, "
      f"lr={best_lr}, epochs={best_ep}")

# Refit best config on full training set
best_mlp = MLPRegressor(hidden_layer_sizes=(best_n,), activation=best_act,
                         solver="adam", learning_rate_init=best_lr,
                         max_iter=best_ep, random_state=RNG)
best_mlp.fit(X_train_s, y_train_s)
mlp_pred_s = best_mlp.predict(X_test_s)
mlp_pred = y_scaler.inverse_transform(mlp_pred_s)

mlp_mse = mean_squared_error(y_test, mlp_pred, multioutput="raw_values")
mlp_r2 = r2_score(y_test, mlp_pred, multioutput="raw_values")

# ---------------------------------------------------------------------
# Random Forest benchmark
# ---------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=300, random_state=RNG)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred, multioutput="raw_values")
rf_r2 = r2_score(y_test, rf_pred, multioutput="raw_values")

# Predictions across the FULL dataset (train+val+test) for plotting --
# metrics above are still computed on the held-out test set only, this
# is purely so the scatter plots show all 600 points, not just the ~120
# in the test split.
X_all_s = x_scaler.transform(X)
mlp_pred_all = y_scaler.inverse_transform(best_mlp.predict(X_all_s))
rf_pred_all = rf.predict(X)
is_test = np.zeros(len(X), dtype=bool)
# recover which rows are the test rows (train_test_split with same random_state)
_, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=RNG)
is_test[test_idx] = True

print("\nTest-set performance (original units):")
print(f"{'Model':<28}{'dP MSE':>12}{'dP R2':>10}{'V MSE':>12}{'V R2':>10}")
print(f"{'MLP('+str(best_n)+','+best_act+')':<28}{mlp_mse[0]:>12.2f}{mlp_r2[0]:>10.4f}{mlp_mse[1]:>12.5f}{mlp_r2[1]:>10.4f}")
print(f"{'RandomForest':<28}{rf_mse[0]:>12.2f}{rf_r2[0]:>10.4f}{rf_mse[1]:>12.5f}{rf_r2[1]:>10.4f}")

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# neuron-count effect at the winning activation/lr/epochs, for context
sub = results_df[(results_df.activation == best_act) &
                  (results_df.lr == best_lr) & (results_df.epochs == best_ep)
                  ].sort_values("neurons")
axes[0].plot(sub.neurons, sub.val_mse, marker="o", color="#2563eb")
axes[0].axvline(best_n, color="#dc2626", linestyle="--", label=f"best n={best_n}")
axes[0].set_xlabel("Hidden-layer neuron count")
axes[0].set_ylabel("Validation MSE (scaled)")
axes[0].set_title(f"Neuron sweep @ {best_act}, lr={best_lr}, epochs={best_ep}")
axes[0].legend()

axes[1].scatter(y[~is_test, 0], mlp_pred_all[~is_test, 0], alpha=0.35, s=22, label="MLP (train)", color="#93b8ec")
axes[1].scatter(y[is_test, 0], mlp_pred_all[is_test, 0], alpha=0.85, s=28, label="MLP (test)", color="#2563eb")
axes[1].scatter(y[~is_test, 0], rf_pred_all[~is_test, 0], alpha=0.35, s=22, label="RF (train)", color="#8fd6a8")
axes[1].scatter(y[is_test, 0], rf_pred_all[is_test, 0], alpha=0.85, s=28, label="RF (test)", color="#16a34a")
lims = [y[:, 0].min(), y[:, 0].max()]
axes[1].plot(lims, lims, "k--", linewidth=1)
axes[1].set_xlabel("True pressure drop [Pa]")
axes[1].set_ylabel("Predicted pressure drop [Pa]")
axes[1].set_title("Pressure drop: predicted vs true (all 600 pts)")
axes[1].legend(fontsize=8)

axes[2].scatter(y[~is_test, 1], mlp_pred_all[~is_test, 1], alpha=0.35, s=22, label="MLP (train)", color="#93b8ec")
axes[2].scatter(y[is_test, 1], mlp_pred_all[is_test, 1], alpha=0.85, s=28, label="MLP (test)", color="#2563eb")
axes[2].scatter(y[~is_test, 1], rf_pred_all[~is_test, 1], alpha=0.35, s=22, label="RF (train)", color="#8fd6a8")
axes[2].scatter(y[is_test, 1], rf_pred_all[is_test, 1], alpha=0.85, s=28, label="RF (test)", color="#16a34a")
lims = [y[:, 1].min(), y[:, 1].max()]
axes[2].plot(lims, lims, "k--", linewidth=1)
axes[2].set_xlabel("True throat velocity [m/s]")
axes[2].set_ylabel("Predicted throat velocity [m/s]")
axes[2].set_title("Velocity: predicted vs true (all 600 pts)")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("model_results.png", dpi=150)
print("\nSaved plot -> model_results.png")

# ---------------------------------------------------------------------
# Export best MLP for JS deployment (single hidden layer forward pass)
# ---------------------------------------------------------------------
export = {
    "features": FEATURES,
    "targets": TARGETS,
    "hidden_neurons": best_n,
    "activation": best_act,
    "learning_rate": best_lr,
    "epochs": best_ep,
    "x_mean": x_scaler.mean_.tolist(),
    "x_scale": x_scaler.scale_.tolist(),
    "y_mean": y_scaler.mean_.tolist(),
    "y_scale": y_scaler.scale_.tolist(),
    "W1": best_mlp.coefs_[0].tolist(),
    "b1": best_mlp.intercepts_[0].tolist(),
    "W2": best_mlp.coefs_[1].tolist(),
    "b2": best_mlp.intercepts_[1].tolist(),
    "test_metrics": {
        "mlp_mse": mlp_mse.tolist(),
        "mlp_r2": mlp_r2.tolist(),
        "rf_mse": rf_mse.tolist(),
        "rf_r2": rf_r2.tolist(),
    },
    "feature_ranges": {
        f: [float(df[f].min()), float(df[f].max())] for f in FEATURES
    },
}
with open("model_export.json", "w") as fh:
    json.dump(export, fh)
print("Exported model -> model_export.json")
