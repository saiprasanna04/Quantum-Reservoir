"""
Quantum Reservoir Computing for Non-Linear Dynamical Systems
Reproducible simulation using a PennyLane state-vector simulator.

This is a simulator experiment, not a physical trapped-ion hardware experiment.
The circuit is trapped-ion-inspired in its use of fixed entangling dynamics;
it does not claim to reproduce a calibrated hardware-specific ion Hamiltonian.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

SEED = 7
N_QUBITS = 4
WINDOW = 4
TRAIN_FRACTION = 0.60
VALID_FRACTION = 0.20
N_POINTS = 500
WASHOUT = 80
RIDGE_ALPHA = 1.0
OUT = Path(__file__).resolve().parents[1]
FIG = OUT / "figures"
RES = OUT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def mackey_glass(n: int = N_POINTS, tau: int = 17, beta: float = 0.2,
                 gamma: float = 0.1, power: int = 10, dt: float = 1.0) -> np.ndarray:
    """Generate a deterministic Mackey–Glass-like series using Euler integration."""
    total = n + WASHOUT + tau + 1
    x = np.zeros(total, dtype=float)
    x[0] = 1.2
    for t in range(1, total):
        delayed = x[t - tau] if t - tau >= 0 else x[0]
        derivative = beta * delayed / (1.0 + delayed ** power) - gamma * x[t - 1]
        x[t] = x[t - 1] + dt * derivative
    series = x[WASHOUT + tau: WASHOUT + tau + n]
    # Stable min-max scaling to the circuit's rotation-angle range.
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo + 1e-12)


def make_windows(series: np.ndarray, window: int = WINDOW):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.asarray(X), np.asarray(y)


def quantum_feature_function(n_qubits: int = N_QUBITS):
    """Return a QNode producing fixed quantum-reservoir observables."""
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def circuit(inputs):
        # Input injection: each lag is mapped to a Y rotation.
        for wire in range(n_qubits):
            qml.RY(np.pi * inputs[wire % len(inputs)], wires=wire)
            qml.RZ(0.35 * np.pi * inputs[(wire + 1) % len(inputs)], wires=wire)

        # Fixed, non-trained reservoir dynamics: local rotations + entanglement.
        for wire in range(n_qubits):
            qml.RX(0.37 * (wire + 1), wires=wire)
        for wire in range(n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
        for wire in range(n_qubits):
            qml.RY(0.23 * (wire + 1), wires=wire)

        # Read out local Z and X observables plus pairwise Z correlations.
        measurements = [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]
        measurements += [qml.expval(qml.PauliX(w)) for w in range(n_qubits)]
        measurements += [
            qml.expval(qml.PauliZ(w) @ qml.PauliZ((w + 1) % n_qubits))
            for w in range(n_qubits)
        ]
        return tuple(measurements)

    return circuit


def quantum_features(X: np.ndarray) -> np.ndarray:
    circuit = quantum_feature_function()
    rows = []
    for row in X:
        values = circuit(row)
        rows.append(np.asarray(values, dtype=float).reshape(-1))
    return np.asarray(rows)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_plot(y_test, pred_dict):
    plt.figure(figsize=(10, 4.8))
    plt.plot(y_test, label="Actual", linewidth=2)
    for label, pred in pred_dict.items():
        plt.plot(pred, label=label, alpha=0.85)
    plt.title("One-step-ahead time-series forecasting")
    plt.xlabel("Test sample")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "forecast_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    names = list(pred_dict)
    values = [np.sqrt(mean_squared_error(y_test, pred_dict[name])) for name in names]
    plt.bar(names, values)
    plt.ylabel("RMSE (lower is better)")
    plt.title("Forecasting error comparison")
    plt.tight_layout()
    plt.savefig(FIG / "rmse_comparison.png", dpi=180)
    plt.close()


def main():
    started = time.perf_counter()
    series = mackey_glass()
    X, y = make_windows(series)
    # Chronological split: no random shuffling to avoid time-series leakage.
    n = len(X)
    train_end = int(n * TRAIN_FRACTION)
    valid_end = int(n * (TRAIN_FRACTION + VALID_FRACTION))
    X_train, y_train = X[:train_end], y[:train_end]
    X_valid, y_valid = X[train_end:valid_end], y[train_end:valid_end]
    X_test, y_test = X[valid_end:], y[valid_end:]

    # Persistence baseline: predict the next value as the last input value.
    persistence_pred = X_test[:, -1]

    # Quantum reservoir feature extraction. Circuit parameters are fixed;
    # only the classical ridge readout is fitted.
    Xq_train = quantum_features(X_train)
    Xq_valid = quantum_features(X_valid)
    Xq_test = quantum_features(X_test)

    scaler = StandardScaler()
    Xq_train_s = scaler.fit_transform(Xq_train)
    Xq_valid_s = scaler.transform(Xq_valid)
    Xq_test_s = scaler.transform(Xq_test)

    # Select alpha from a small, explicit validation set.
    candidate_alphas = [0.01, 0.1, 1.0, 10.0]
    best_alpha, best_rmse = None, float("inf")
    for alpha in candidate_alphas:
        model = Ridge(alpha=alpha)
        model.fit(Xq_train_s, y_train)
        val_pred = model.predict(Xq_valid_s)
        rmse = np.sqrt(mean_squared_error(y_valid, val_pred))
        if rmse < best_rmse:
            best_rmse, best_alpha = float(rmse), float(alpha)

    # Refit on train + validation after hyperparameter selection.
    Xq_fit = np.vstack([Xq_train, Xq_valid])
    y_fit = np.concatenate([y_train, y_valid])
    scaler_final = StandardScaler().fit(Xq_fit)
    Xq_fit_s = scaler_final.transform(Xq_fit)
    Xq_test_s = scaler_final.transform(Xq_test)
    qrc_model = Ridge(alpha=best_alpha).fit(Xq_fit_s, y_fit)
    qrc_pred = qrc_model.predict(Xq_test_s)

    # Classical comparator: ridge regression on the same lagged inputs.
    classical_scaler = StandardScaler().fit(np.vstack([X_train, X_valid]))
    Xc_fit = classical_scaler.transform(np.vstack([X_train, X_valid]))
    Xc_test = classical_scaler.transform(X_test)
    classical_model = Ridge(alpha=best_alpha).fit(
        Xc_fit, np.concatenate([y_train, y_valid])
    )
    classical_pred = classical_model.predict(Xc_test)

    predictions = {
        "Persistence": persistence_pred,
        "Classical Ridge": classical_pred,
        "Quantum Reservoir": qrc_pred,
    }
    metrics = {
        name: regression_metrics(y_test, pred)
        for name, pred in predictions.items()
    }
    elapsed = time.perf_counter() - started
    report = {
        "project_title": "Quantum Reservoir Computing for Non-Linear Dynamical Systems: Time-Series Forecasting on Trapped-Ion Simulators",
        "dataset": "Synthetic Mackey–Glass-like series generated by the script",
        "n_points": int(len(series)),
        "window_size": WINDOW,
        "n_qubits": N_QUBITS,
        "train_validation_test_split": "60% / 20% / 20% chronological",
        "best_ridge_alpha": best_alpha,
        "validation_rmse": best_rmse,
        "metrics": metrics,
        "runtime_seconds": float(elapsed),
        "important_scope_note": "Ideal PennyLane state-vector simulation; trapped-ion-inspired circuit, not physical trapped-ion hardware.",
    }
    (RES / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_plot(y_test, predictions)
    print(json.dumps(report, indent=2))
    print(f"\nSaved plots to: {FIG}")
    print(f"Saved metrics to: {RES / 'results.json'}")


if __name__ == "__main__":
    main()
