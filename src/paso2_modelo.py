"""
Paso 2 - Modelo Base
Preparación para superficie de decisión y curva de aprendizaje
"""

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_data(path="data/paso1_features.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("No se encontró el dataset de Paso 1.")
    return pd.read_csv(path)


def prepare_model_data(df):
    """
    Selecciona features numéricas relevantes para clasificación.
    """
    features = ["price_smooth", "amount", "avg_amount_48h", "segment_te", "volume"]
    target = "is_suspicious"

    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def train_logistic_regression(X, y):
    """
    Entrena modelo base de clasificación.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy

def decision_surface_2d(df, out_path="reports/paso2_decision_surface.png"):
    Path("reports").mkdir(exist_ok=True)

    features = ["price_smooth", "amount", "avg_amount_48h", "segment_te", "volume"]
    target = "is_suspicious"

    df = df[features + [target]].dropna().copy()
    X = df[features].to_numpy()
    y = df[target].to_numpy()

    # Escalar
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Reducir a 2D con PCA
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xs)

    # Entrenar modelo en 2D (más fácil para graficar)
    model = LogisticRegression(max_iter=1000)
    model.fit(X2, y)

    # Crear grid
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, probs, levels=20, alpha=0.7)
    plt.colorbar(label="Probabilidad de sospechoso")

    # Para que no sea enorme, muestreamos puntos
    idx = np.random.default_rng(42).choice(len(X2), size=min(3000, len(X2)), replace=False)
    plt.scatter(X2[idx, 0], X2[idx, 1], c=y[idx], s=10, alpha=0.6)

    plt.title("Superficie de decisión (PCA 2D + Logistic Regression)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"✅ Decision surface guardada en: {out_path}")


def main():
    print("Paso 2 - Modelo base iniciado")

    df = load_data()
    print("Datos cargados:", df.shape)

    X, y, scaler = prepare_model_data(df)
    model, acc = train_logistic_regression(X, y)

    print(f"Accuracy del modelo base: {acc:.4f}")
    
    decision_surface_2d(df)

if __name__ == "__main__":
    main()