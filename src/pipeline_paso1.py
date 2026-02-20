"""
Paso 1: El Caos de los Datos (Pandas, NumPy & SciPy)
Pipeline base (se irá completando por commits).
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter


def generate_synthetic_data(n_users=5000, n_tx=200000, seed=42):
    rng = np.random.default_rng(seed)

    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "segment": rng.integers(1, 2000, size=n_users),  # alta cardinalidad
        "country": rng.choice(["PE", "CL", "CO", "MX"], size=n_users)
    })

    start = pd.Timestamp("2026-02-10")
    timestamps = start + pd.to_timedelta(
        rng.integers(0, 10 * 24 * 3600, size=n_tx),
        unit="s"
    )

    transactions = pd.DataFrame({
        "tx_id": np.arange(1, n_tx + 1),
        "user_id": rng.integers(1, n_users + 1, size=n_tx),
        "timestamp": timestamps,
        "price": rng.normal(100, 25, size=n_tx).clip(1),
        "volume": rng.integers(1, 5, size=n_tx),
        "is_suspicious": rng.choice([0, 1], size=n_tx, p=[0.99, 0.01]),
    })

    return users, transactions


def main():
    print("✅ Paso 1 iniciado")
    print("Versiones:")
    print("  pandas:", pd.__version__)
    print("  numpy:", np.__version__)

    users, tx = generate_synthetic_data()
    print("Users:", users.shape, "Tx:", tx.shape)


if __name__ == "__main__":
    main()