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


def build_features_pandas(users: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])

    df = tx.merge(users, on="user_id", how="left", validate="many_to_one")
    df["amount"] = df["price"] * df["volume"]
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def rolling_avg_48h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Crear avg_amount_48h por usuario SIN apply:
    # 1) ponemos timestamp como índice temporal
    # 2) hacemos rolling por tiempo y calculamos la media del amount
    out = (
        df.set_index("timestamp")
          .groupby("user_id")["amount"]
          .rolling("48h")
          .mean()
          .reset_index(name="avg_amount_48h")
    )

    # out tiene columnas: user_id, timestamp, avg_amount_48h
    # lo unimos de vuelta con df por user_id y timestamp
    df = df.merge(out, on=["user_id", "timestamp"], how="left")
    return df
    
def target_encoding_numpy(df: pd.DataFrame, cat_col="segment", target_col="is_suspicious") -> pd.DataFrame:
    df = df.copy()

    # Convertimos categoría a códigos 0..k-1 sin for
    cats, inv = np.unique(df[cat_col].to_numpy(), return_inverse=True)

    y = df[target_col].astype(float).to_numpy()

    # suma del target por categoría
    sums = np.bincount(inv, weights=y)
    # conteo por categoría
    counts = np.bincount(inv)

    means = sums / np.maximum(counts, 1)

    # asignar encoding a cada fila
    df[f"{cat_col}_te"] = means[inv]
    return df

def main():
    print("✅ Paso 1 iniciado")
    print("Versiones:")
    print("  pandas:", pd.__version__)
    print("  numpy:", np.__version__)

    users, tx = generate_synthetic_data()
    print("Users:", users.shape, "Tx:", tx.shape)

    df = build_features_pandas(users, tx)
    df = rolling_avg_48h(df)

    df = target_encoding_numpy(df, cat_col="segment", target_col="is_suspicious")
    print("Target Encoding listo. Columnas nuevas:", ["segment_te"])
    print(df[["segment", "segment_te", "is_suspicious"]].head())

    print("DF con features (pandas):", df.shape)
    print(df[["user_id", "timestamp", "amount", "avg_amount_48h"]].head())
    

if __name__ == "__main__":
    main()