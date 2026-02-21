"""
Paso 2 - Modelo Base
Preparación para superficie de decisión y curva de aprendizaje
"""

import pandas as pd
import numpy as np
from pathlib import Path

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


def main():
    print("Paso 2 - Modelo base iniciado")

    df = load_data()
    print("Datos cargados:", df.shape)

    X, y, scaler = prepare_model_data(df)
    model, acc = train_logistic_regression(X, y)

    print(f"Accuracy del modelo base: {acc:.4f}")


if __name__ == "__main__":
    main()