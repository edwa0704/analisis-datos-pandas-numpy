"""
Paso 2 - Modelo y Visualización Avanzada (Matplotlib)
Incluye:
- Modelo base (Logistic Regression)
- Superficie de decisión en 2D (PCA + Logistic Regression)
- Curva de aprendizaje Loss vs Epochs (SGDClassifier)
- Dashboard final 2x2 (Matplotlib subplots)
- Opción --show: abre automáticamente las imágenes generadas
"""

import time
import argparse
import os
import sys
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # evita errores de Tkinter / GUI
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ---------------------------
# CLI (argumentos)
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Paso 2 - Modelo + Dashboard")
    p.add_argument("--show", action="store_true", help="Abrir imágenes al finalizar")
    return p.parse_args()


# ---------------------------
# Abrir archivo con visor del sistema
# ---------------------------
def open_file(path: str):
    try:
        p = Path(path).resolve()
        if not p.exists():
            print(f"⚠ Archivo no existe: {p}")
            return

        if sys.platform.startswith("win"):
            os.startfile(str(p))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=False)
        else:
            subprocess.run(["xdg-open", str(p)], check=False)

    except Exception as e:
        print("⚠ No se pudo abrir automáticamente:", e)


def load_data(path="data/paso1_features.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "No se encontró el dataset de Paso 1.\n"
            "Ejecuta primero: python src/pipeline_paso1.py"
        )
    return pd.read_csv(path)


def prepare_model_data(df):
    features = ["price_smooth", "amount", "avg_amount_48h", "segment_te", "volume"]
    target = "is_suspicious"

    df = df[features + [target]].dropna().copy()

    X = df[features]
    y = df[target].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy


def decision_surface_2d(df, out_path="reports/paso2_decision_surface.png"):
    Path("reports").mkdir(exist_ok=True)

    features = ["price_smooth", "amount", "avg_amount_48h", "segment_te", "volume"]
    target = "is_suspicious"

    work = df[features + [target]].dropna().copy()
    X = work[features].to_numpy()
    y = work[target].to_numpy().astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)

    model = LogisticRegression(max_iter=1000)
    model.fit(X2, y)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, probs, levels=20, alpha=0.7)
    plt.colorbar(label="Probabilidad de sospechoso")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X2), size=min(3000, len(X2)), replace=False)
    plt.scatter(X2[idx, 0], X2[idx, 1], c=y[idx], s=10, alpha=0.6)

    plt.title("Superficie de decisión (PCA 2D + Logistic Regression)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    out_abs = str(Path(out_path).resolve())
    print(f"✅ Decision surface guardada en: {out_abs}")
    return out_abs


def learning_curve_loss_epochs(df, epochs=20, sample=50000, out_path="reports/paso2_learning_curve.png"):
    Path("reports").mkdir(exist_ok=True)

    features = ["price_smooth", "amount", "avg_amount_48h", "segment_te", "volume"]
    target = "is_suspicious"

    work = df[features + [target]].dropna().copy()

    if len(work) > sample:
        work = work.sample(n=sample, random_state=42)

    X = work[features].to_numpy()
    y = work[target].to_numpy().astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        random_state=42
    )

    classes = np.array([0, 1], dtype=int)
    losses = []

    for epoch in range(1, epochs + 1):
        clf.partial_fit(Xs, y, classes=classes)
        proba = clf.predict_proba(Xs)
        loss = log_loss(y, proba, labels=classes)
        losses.append(loss)
        print(f"Epoch {epoch}/{epochs} - loss={loss:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), losses, marker="o")
    plt.title("Curva de aprendizaje - Loss vs Epochs (SGDClassifier)")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    out_abs = str(Path(out_path).resolve())
    print(f"✅ Learning curve guardada en: {out_abs}")
    return losses, out_abs


def build_final_2x2_figure(df, losses, out_path="reports/paso2_2x2_dashboard.png"):
    Path("reports").mkdir(exist_ok=True)

    features = ["price_smooth", "amount", "avg_amount_48h", "segment_te", "volume"]
    target = "is_suspicious"

    work = df[features + [target]].dropna().copy()
    X = work[features].to_numpy()
    y = work[target].to_numpy().astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)

    model = LogisticRegression(max_iter=1000)
    model.fit(X2, y)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X2), size=min(3000, len(X2)), replace=False)

    axes[0, 0].scatter(X2[idx, 0], X2[idx, 1], c=y[idx], s=10, alpha=0.6)
    axes[0, 0].set_title("PCA Scatter (clases 0/1)")
    axes[0, 0].set_xlabel("PCA 1")
    axes[0, 0].set_ylabel("PCA 2")

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    axes[0, 1].contourf(xx, yy, probs, levels=20, alpha=0.7)
    axes[0, 1].set_title("Decision Surface (probabilidad)")
    axes[0, 1].set_xlabel("PCA 1")
    axes[0, 1].set_ylabel("PCA 2")

    axes[1, 0].plot(range(1, len(losses) + 1), losses, marker="o")
    axes[1, 0].set_title("Loss vs Epochs")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Log Loss")
    axes[1, 0].grid(True, alpha=0.3)

    normal = df[df["is_suspicious"] == 0]["price"].dropna()
    suspicious = df[df["is_suspicious"] == 1]["price"].dropna()

    axes[1, 1].hist(normal, bins=50, alpha=0.6, label="Normal")
    axes[1, 1].hist(suspicious, bins=50, alpha=0.6, label="Sospechosa")
    axes[1, 1].set_title("Distribución de Precio")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    out_abs = str(Path(out_path).resolve())
    print(f"✅ Figura 2x2 guardada en: {out_abs}")
    return out_abs


def main():
    args = parse_args()

    print("📌 Working directory:", Path.cwd())
    print("Paso 2 - Modelo base iniciado")

    df = load_data()
    print("Datos cargados:", df.shape)

    X, y, _ = prepare_model_data(df)
    _, acc = train_logistic_regression(X, y)
    print(f"Accuracy del modelo base: {acc:.4f}")

    # Generar imágenes
    path_surface = decision_surface_2d(df)
    losses, path_curve = learning_curve_loss_epochs(df, epochs=20, sample=50000)
    path_dashboard = build_final_2x2_figure(df, losses)

    print("Abrir surface:", path_surface)
    print("Abrir curve:", path_curve)
    print("Abrir dashboard:", path_dashboard)

    # Abrir con pausas para que Windows no se salte la tercera
    if args.show:
        open_file(path_surface)
        time.sleep(0.8)

        open_file(path_curve)
        time.sleep(0.8)

        open_file(path_dashboard)
        time.sleep(0.8)


if __name__ == "__main__":
    main()