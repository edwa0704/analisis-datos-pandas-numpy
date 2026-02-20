"""
Paso 1: El Caos de los Datos (Pandas, NumPy & SciPy)
Pipeline base (se irá completando por commits).
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
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

    s = (
        df.set_index("timestamp")
          .groupby("user_id")["amount"]
          .rolling("48h")
          .mean()
    )

    # s es una Serie con MultiIndex (user_id, timestamp) y está en el mismo orden que df
    # Lo convertimos a numpy y lo asignamos por orden
    df["avg_amount_48h"] = s.to_numpy()
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

def smooth_price_savgol(df: pd.DataFrame, col="price", window=51, poly=3) -> pd.DataFrame:
    df = df.copy()

    # Ordenar por tiempo para que la tendencia tenga sentido
    df = df.sort_values("timestamp").reset_index(drop=True)

    y = df[col].to_numpy()

    # Ajustar ventana (debe ser impar y menor o igual al tamaño)
    n = len(y)
    if n < 7:
        df[f"{col}_smooth"] = y
        return df

    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < 5:
        window = 5
    if window % 2 == 0:
        window += 1

    df[f"{col}_smooth"] = savgol_filter(y, window_length=window, polyorder=poly)
    return df

def ks_test_normal_vs_suspicious(df: pd.DataFrame, col="price"):
    normal = df.loc[df["is_suspicious"] == 0, col].to_numpy()
    suspicious = df.loc[df["is_suspicious"] == 1, col].to_numpy()

    # SciPy KS test: compara distribuciones
    stat, p_value = stats.ks_2samp(normal, suspicious)
    return stat, p_value, len(normal), len(suspicious)

def save_outputs(df: pd.DataFrame, output_dir="data"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guardar dataset final (solo algunas columnas para que no pese demasiado)
    cols_to_save = [
        "tx_id", "user_id", "timestamp", "price", "price_smooth", "volume",
        "amount", "avg_amount_48h", "segment", "segment_te", "country", "is_suspicious"
    ]
    existing_cols = [c for c in cols_to_save if c in df.columns]
    df[existing_cols].to_csv(out_dir / "paso1_features.csv", index=False)

    print(f"✅ Archivo guardado: {out_dir / 'paso1_features.csv'}")


def save_report(text: str, output_dir="reports", filename="paso1_reporte.txt"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / filename
    report_path.write_text(text, encoding="utf-8")

    print(f"✅ Reporte guardado: {report_path}")

def data_quality_checks(df: pd.DataFrame) -> dict:
    checks = {}

    checks["rows"] = int(df.shape[0])
    checks["cols"] = int(df.shape[1])

    # Nulos por columna (top 10)
    nulls = df.isna().sum().sort_values(ascending=False)
    checks["nulls_top10"] = nulls.head(10).to_dict()

    # Duplicados (por tx_id debería ser 0)
    if "tx_id" in df.columns:
        checks["dup_tx_id"] = int(df["tx_id"].duplicated().sum())

    # Rango simple de price/volume
    if "price" in df.columns:
        checks["price_min"] = float(df["price"].min())
        checks["price_max"] = float(df["price"].max())
    if "volume" in df.columns:
        checks["volume_min"] = int(df["volume"].min())
        checks["volume_max"] = int(df["volume"].max())

    return checks


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Enteros (reduce memoria)
    int_cols = ["tx_id", "user_id", "segment", "volume", "is_suspicious"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], downcast="integer")

    # Floats
    float_cols = ["price", "price_smooth", "amount", "avg_amount_48h", "segment_te"]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], downcast="float")

    # Categorías
    cat_cols = ["country"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Paso 1 - Pipeline de features")
    parser.add_argument("--n_users", type=int, default=5000)
    parser.add_argument("--n_tx", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    # 1️⃣ Parsear argumentos
    args = parse_args()

    print("✅ Paso 1 iniciado")
    print("Versiones:")
    print("  pandas:", pd.__version__)
    print("  numpy:", np.__version__)
    print(f"Parametros: n_users={args.n_users}, n_tx={args.n_tx}, seed={args.seed}")

    # 2️⃣ Generar datos con argumentos
    users, tx = generate_synthetic_data(
        n_users=args.n_users,
        n_tx=args.n_tx,
        seed=args.seed
    )
    print("Users:", users.shape, "Tx:", tx.shape)

    df = build_features_pandas(users, tx)
    df = rolling_avg_48h(df)

    df = target_encoding_numpy(df, cat_col="segment", target_col="is_suspicious")
    print("Target Encoding listo. Columnas nuevas:", ["segment_te"])
    print(df[["segment", "segment_te", "is_suspicious"]].head())
    
    df = smooth_price_savgol(df, col="price", window=51, poly=3)
    print("Suavizado Savitzky-Golay listo. Columnas nuevas:", ["price_smooth"])
    print(df[["timestamp", "price", "price_smooth"]].head())

    stat, p_value, n_normal, n_susp = ks_test_normal_vs_suspicious(df, col="price")
    print("KS test (price) normal vs sospechosa:")
    print(f"  n_normal={n_normal}, n_suspicious={n_susp}")
    print(f"  KS_statistic={stat:.6f}, p_value={p_value:.6e}")

    # Optimización de memoria / tipos
    df = optimize_dtypes(df)

    # Checks de calidad
    checks = data_quality_checks(df)
    print("✅ Data Quality Checks (resumen):")
    print(checks)

    print("DF con features (pandas):", df.shape)
    print(df[["user_id", "timestamp", "amount", "avg_amount_48h"]].head())
    
    # Reporte resumen
    report = []
    report.append("Paso 1 - Resumen\n")
    report.append(f"Users: {users.shape}\n")
    report.append(f"Tx: {tx.shape}\n")
    report.append(f"DF final: {df.shape}\n")
    report.append(f"Columnas: {df.columns.tolist()}\n")
    report.append("\nKS test (price) normal vs sospechosa\n")
    report.append(f"n_normal={n_normal}, n_suspicious={n_susp}\n")
    report.append(f"KS_statistic={stat:.6f}, p_value={p_value:.6e}\n")
    report.append("\nData Quality Checks (top):\n")
    report.append(f"{checks}\n")

    save_outputs(df, output_dir="data")
    save_report("".join(report), output_dir="reports", filename="paso1_reporte.txt")

if __name__ == "__main__":
    main()