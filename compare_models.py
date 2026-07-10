"""
Hyperparameter comparison study for the CFD surrogate model.

Compares, on the same train/val/test split:
  - activation function: ReLU vs tanh (single hidden layer, per project spec)
  - epoch count: how training length affects validation error for each activation
  - hidden-neuron count: restricted to the 0-25 range
  - MLP (best config) vs Random Forest benchmark

Produces:
  - comparison_activation_epochs.png  (MSE curves: neurons x epochs, relu vs tanh)
  - comparison_mlp_vs_rf.png          (final model comparison, both outputs)
  - comparison_summary.csv            (full grid results, sortable)
  - prints the winning configuration and its test-set metrics
"""
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
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RNG = 23
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
# Grid: neurons restricted to 0-25, activation relu/tanh, several epoch
# budgets. Learning rate fixed at 0.05 (found best in the earlier sweep)
# so the comparison isolates activation + epochs + neuron count.
# ---------------------------------------------------------------------
neuron_grid = [2, 4, 6, 8, 10, 12, 15, 18, 20, 22, 25]
activations = ["relu", "tanh"]
epoch_grid = [100, 300, 500, 1000]
LR = 0.05

records = []
for act in activations:
    for ep in epoch_grid:
        for n in neuron_grid:
            mlp = MLPRegressor(hidden_layer_sizes=(n,), activation=act,
                                solver="adam", learning_rate_init=LR,
                                max_iter=ep, random_state=RNG)
            mlp.fit(X_tr, y_tr)
            pred = mlp.predict(X_val)
            mse = mean_squared_error(y_val, pred)
            records.append({"activation": act, "epochs": ep, "neurons": n,
                             "val_mse": mse})

grid_df = pd.DataFrame(records)
grid_df.to_csv("comparison_summary.csv", index=False)

best_row = grid_df.loc[grid_df.val_mse.idxmin()]
print("=== Best configuration overall (neurons restricted to 0-25) ===")
print(best_row.to_string())

# ---------------------------------------------------------------------
# Plot 1: MSE vs neuron count, one panel per epoch budget, relu vs tanh
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, len(epoch_grid), figsize=(4.2 * len(epoch_grid), 4), sharey=True)
colors = {"relu": "#2563eb", "tanh": "#dc2626"}
for ax, ep in zip(axes, epoch_grid):
    for act in activations:
        sub = grid_df[(grid_df.epochs == ep) & (grid_df.activation == act)].sort_values("neurons")
        ax.plot(sub.neurons, sub.val_mse, marker="o", color=colors[act], label=act)
    ax.set_title(f"epochs = {ep}")
    ax.set_xlabel("hidden neurons (0-25)")
    if best_row.epochs == ep:
        ax.axvline(best_row.neurons, color="gray", linestyle=":")
axes[0].set_ylabel("Validation MSE (scaled)")
axes[0].legend()
plt.suptitle("ReLU vs tanh: validation MSE across neuron count and training epochs")
plt.tight_layout()
plt.savefig("comparison_activation_epochs.png", dpi=150)
print("Saved -> comparison_activation_epochs.png")

# ---------------------------------------------------------------------
# Refit the winning MLP config on full training data, compare vs RF
# ---------------------------------------------------------------------
best_mlp = MLPRegressor(hidden_layer_sizes=(int(best_row.neurons),),
                         activation=best_row.activation, solver="adam",
                         learning_rate_init=LR, max_iter=int(best_row.epochs),
                         random_state=RNG)
best_mlp.fit(X_train_s, y_train_s)
mlp_pred = y_scaler.inverse_transform(best_mlp.predict(X_test_s))
mlp_mse = mean_squared_error(y_test, mlp_pred, multioutput="raw_values")
mlp_r2 = r2_score(y_test, mlp_pred, multioutput="raw_values")

rf = RandomForestRegressor(n_estimators=300, random_state=RNG)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred, multioutput="raw_values")
rf_r2 = r2_score(y_test, rf_pred, multioutput="raw_values")

print("\n=== Test-set comparison: tuned MLP vs Random Forest ===")
label = f"MLP({int(best_row.neurons)},{best_row.activation},ep={int(best_row.epochs)})"
print(f"{'Model':<30}{'dP MSE':>12}{'dP R2':>10}{'V MSE':>12}{'V R2':>10}")
print(f"{label:<30}{mlp_mse[0]:>12.2f}{mlp_r2[0]:>10.4f}{mlp_mse[1]:>12.5f}{mlp_r2[1]:>10.4f}")
print(f"{'RandomForest':<30}{rf_mse[0]:>12.2f}{rf_r2[0]:>10.4f}{rf_mse[1]:>12.5f}{rf_r2[1]:>10.4f}")

# ---------------------------------------------------------------------
# Plot 2: side-by-side bar comparison, both outputs, both metrics
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
models = [label, "RandomForest"]

dp_mse = [mlp_mse[0], rf_mse[0]]
v_mse = [mlp_mse[1], rf_mse[1]]
axes[0].bar(models, dp_mse, color=["#2563eb", "#16a34a"])
axes[0].set_title("Pressure-drop MSE (Pa\u00b2)")
axes[0].tick_params(axis='x', rotation=15)

dp_r2 = [mlp_r2[0], rf_r2[0]]
v_r2 = [mlp_r2[1], rf_r2[1]]
x = np.arange(2)
width = 0.35
axes[1].bar(x - width/2, dp_r2, width, label="\u0394P R\u00b2", color="#e2a23b")
axes[1].bar(x + width/2, v_r2, width, label="V R\u00b2", color="#4fd6d0")
axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=15)
axes[1].set_ylim(0.8, 1.0)
axes[1].set_title("R\u00b2 by output")
axes[1].legend()

plt.tight_layout()
plt.savefig("comparison_mlp_vs_rf.png", dpi=150)
print("Saved -> comparison_mlp_vs_rf.png")
