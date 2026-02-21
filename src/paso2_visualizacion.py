"""
Paso 2: Visualización de Alta Dimensionalidad (Matplotlib & Seaborn)
- Reduce dimensionalidad a 2D con t-SNE (scikit-learn)
- Grafica con Seaborn scatterplot:
    color = price
    size  = volume
- Siempre guarda la imagen en reports/
- Opcional: abrir automáticamente la imagen con --show
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # ✅ backend sin GUI (evita errores de Tkinter)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def parse_args():
    p = argparse.ArgumentParser(description="Paso 2 - t-SNE visualización")
    p.add_argument("--input", default="data/paso1_features.csv", help="CSV generado en Paso 1")
    p.add_argument("--sample", type=int, default=10000, help="cantidad de filas para t-SNE (recomendado 5k-20k)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perplexity", type=float, default=30)
    p.add_argument("--show", action="store_true", help="Abrir la imagen al finalizar (si el sistema lo permite)")
    return p.parse_args()


def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "No se encontró el CSV de Paso 1.\n"
            "Ejecuta primero: python src/pipeline_paso1.py\n"
            "Luego:         python src/paso2_visualizacion.py --sample 10000"
        )
    return pd.read_csv(path)


def reduce_dimensionality(df: pd.DataFrame, sample=10000, seed=42, perplexity=30) -> pd.DataFrame:
    features = ["price_smooth", "amount", "avg_amount_48h", "segment_te", "volume"]
    cols_needed = features + ["price"]  # price se usa para color, volume ya está en features

    # eliminar duplicados por seguridad
    cols_needed = list(dict.fromkeys(cols_needed))

    work = df[cols_needed].dropna().copy()

    # Debug: detectar columnas duplicadas (por seguridad)
    if work.columns.duplicated().any():
        print("⚠ Columnas duplicadas detectadas:", work.columns[work.columns.duplicated()].tolist())
        work = work.loc[:, ~work.columns.duplicated()]

    # Muestreo para t-SNE (t-SNE es lento con 200k)
    if len(work) > sample:
        work = work.sample(n=sample, random_state=seed)

    X = work[features].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto"
    )

    components = tsne.fit_transform(X_scaled)

    work["comp1"] = components[:, 0].astype(float)
    work["comp2"] = components[:, 1].astype(float)

    return work


def open_image_with_system_viewer(path: Path):
    """Intenta abrir la imagen con el visor del sistema (Windows/Mac/Linux)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # Windows: Fotos
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print("⚠ No se pudo abrir automáticamente la imagen, pero ya está guardada en reports/.")
        print("Detalle:", e)


def plot_tsne(df: pd.DataFrame, show: bool = False) -> Path:
    Path("reports").mkdir(exist_ok=True)

    plot_df = df[["comp1", "comp2", "price", "volume"]].copy()

    # Forzar a 1D numérico (evita dtype object/arrays)
    plot_df["comp1"] = pd.to_numeric(plot_df["comp1"], errors="coerce")
    plot_df["comp2"] = pd.to_numeric(plot_df["comp2"], errors="coerce")
    plot_df["price"] = pd.to_numeric(plot_df["price"], errors="coerce")
    plot_df["volume"] = pd.to_numeric(plot_df["volume"], errors="coerce")
    plot_df = plot_df.dropna()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="comp1",
        y="comp2",
        hue="price",       # color representa precio
        size="volume",     # tamaño representa volumen
        sizes=(20, 200),
        alpha=0.7
    )
    plt.title("t-SNE - Visualización de Transacciones")
    plt.tight_layout()

    out_path = Path("reports/paso2_tsne_scatter.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"✅ Imagen guardada en: {out_path}")

    # Mostrar (abrir archivo) solo si el usuario lo pide
    if show:
        open_image_with_system_viewer(out_path)

    return out_path


def main():
    args = parse_args()
    print("Paso 2 iniciado...")
    df = load_data(args.input)
    print("Datos cargados:", df.shape)

    df_reduced = reduce_dimensionality(
        df,
        sample=args.sample,
        seed=args.seed,
        perplexity=args.perplexity
    )
    print("Muestra usada para t-SNE:", df_reduced.shape)

    plot_tsne(df_reduced, show=args.show)

    # Debug opcional (deja evidencia en consola)
    print(df_reduced[["comp1", "comp2", "price", "volume"]].head())
    print(df_reduced[["comp1", "comp2", "price", "volume"]].dtypes)


if __name__ == "__main__":
    main()