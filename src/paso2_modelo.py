"""
Paso 2 - Modelo y Visualización Avanzada (Matplotlib)
Incluye:
- Modelo base (Logistic Regression)
- Superficie de decisión en 2D (PCA + Logistic Regression)
- Curva de aprendizaje Loss vs Epochs (SGDClassifier)
- Dashboard final 2x2 (Matplotlib subplots)
- Opción --show: abre automáticamente las imágenes generadas
"""

import imageio.v2 as imageio
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

    p.add_argument("--epochs", type=int, default=20, help="Cantidad de épocas para SGDClassifier (loss vs epochs)")
    p.add_argument("--sample", type=int, default=50000, help="Muestra de filas para curva de aprendizaje (rendimiento)")
    p.add_argument("--open-delay", type=float, default=0.8, help="Pausa (segundos) entre aperturas en Windows")

    # por defecto abre en Windows, si no quieres: --no-show
    p.add_argument("--no-show", action="store_true", help="No abrir imágenes/reportes al finalizar")

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


def learning_curve_realtime_gif(df, epochs=20, sample=50000, gif_path="reports/paso2_learning_curve_realtime.gif"):
    """
    Genera un GIF que muestra la curva Loss vs Epochs actualizándose en cada época.
    No requiere GUI (usa Agg) y cumple el 'tiempo real' como evidencia visual.
    """
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

    frames = []
    tmp_dir = Path("reports/_frames")
    tmp_dir.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        clf.partial_fit(Xs, y, classes=classes)

        proba = clf.predict_proba(Xs)
        loss = log_loss(y, proba, labels=classes)
        losses.append(loss)

        # generar frame
        fig = plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(losses) + 1), losses, marker="o")
        plt.title("Loss vs Epochs (Tiempo real)")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        frame_path = tmp_dir / f"frame_{epoch:03d}.png"
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)

        frames.append(imageio.imread(frame_path))
        print(f"Epoch {epoch}/{epochs} - loss={loss:.6f}")

    imageio.mimsave(gif_path, frames, duration=0.35)
    print(f"✅ GIF guardado en: {Path(gif_path).resolve()}")

    return losses, str(Path(gif_path).resolve())


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

def save_step2_report(acc, losses, path_surface, path_curve, path_dashboard, filename="reports/paso2_reporte.txt"):
    Path("reports").mkdir(exist_ok=True)

    def line(k, v, w=22):
        return f"{k:<{w}}: {v}\n"

    report = []
    report.append("=" * 50 + "\n")
    report.append("PASO 2 - REPORTE (MODELADO + VISUALIZACIÓN)\n")
    report.append("=" * 50 + "\n\n")

    report.append("[EJECUCIÓN]\n")
    report.append(line("Working directory", Path.cwd()))
    report.append("\n")

    report.append("[MODELO BASE]\n")
    report.append(line("Modelo", "Logistic Regression"))
    report.append(line("Accuracy", f"{acc:.4f}"))
    report.append("\n")

    report.append("[APRENDIZAJE]\n")
    report.append(line("Modelo", "SGDClassifier (log_loss)"))
    report.append(line("Epochs", len(losses)))
    if losses:
        report.append(line("Loss inicial", f"{losses[0]:.6f}"))
        report.append(line("Loss final", f"{losses[-1]:.6f}"))
    report.append("\n")

    report.append("[SALIDAS GENERADAS]\n")
    report.append(line("Decision surface", Path(path_surface).name))
    report.append(line("Learning curve", Path(path_curve).name))
    report.append(line("Dashboard 2x2", Path(path_dashboard).name))
    report.append("\n")

    report.append("[RUTAS COMPLETAS]\n")
    report.append(line("Decision surface", path_surface))
    report.append(line("Learning curve", path_curve))
    report.append(line("Dashboard 2x2", path_dashboard))

    out_path = Path(filename)
    out_path.write_text("".join(report), encoding="utf-8")
    print(f"✅ Reporte Paso 2 guardado en: {out_path.resolve()}")
    return str(out_path.resolve())

def save_step2_report_html(acc, losses, path_surface, path_curve, path_dashboard, filename="reports/paso2_reporte.html"):
    Path("reports").mkdir(exist_ok=True)

    loss_ini = f"{losses[0]:.6f}" if losses else "N/A"
    loss_fin = f"{losses[-1]:.6f}" if losses else "N/A"

    html = f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>Paso 2 - Reporte</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin: 0 0 8px 0; }}
    .muted {{ color: #555; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 14px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
    th {{ background: #f6f6f6; }}
    code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>PASO 2 - REPORTE (Modelado + Visualización)</h1>
  <div class="muted">Working directory: <code>{Path.cwd()}</code></div>

  <div class="card">
    <h2>Modelo base</h2>
    <table>
      <tr><th>Modelo</th><td>Logistic Regression</td></tr>
      <tr><th>Accuracy</th><td>{acc:.4f}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Aprendizaje</h2>
    <table>
      <tr><th>Modelo</th><td>SGDClassifier (log_loss)</td></tr>
      <tr><th>Epochs</th><td>{len(losses)}</td></tr>
      <tr><th>Loss inicial</th><td>{loss_ini}</td></tr>
      <tr><th>Loss final</th><td>{loss_fin}</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Salidas generadas</h2>
    <table>
      <tr><th>Decision surface</th><td><code>{Path(path_surface).name}</code></td></tr>
      <tr><th>Learning curve</th><td><code>{Path(path_curve).name}</code></td></tr>
      <tr><th>Dashboard 2x2</th><td><code>{Path(path_dashboard).name}</code></td></tr>
    </table>
    <p class="muted">Rutas completas:</p>
    <ul>
      <li>{path_surface}</li>
      <li>{path_curve}</li>
      <li>{path_dashboard}</li>
    </ul>
  </div>
</body>
</html>
"""
    out = Path(filename)
    out.write_text(html, encoding="utf-8")
    print(f"✅ Reporte HTML guardado en: {out.resolve()}")
    return str(out.resolve())

def main():
    args = parse_args()

    print("📌 Working directory:", Path.cwd())
    print("Paso 2 - Modelo base iniciado")

    df = load_data()
    print("Datos cargados:", df.shape)

    X, y, _ = prepare_model_data(df)
    _, acc = train_logistic_regression(X, y)
    print(f"Accuracy del modelo base: {acc:.4f}")

    # 1️⃣ Superficie de decisión
    path_surface = decision_surface_2d(df)

    # 2️⃣ Curva en tiempo real (GIF)
    losses, gif_path = learning_curve_realtime_gif(
        df,
        epochs=args.epochs,
        sample=args.sample
    )

    # 3️⃣ Dashboard 2x2 (usa losses del GIF)
    path_dashboard = build_final_2x2_figure(df, losses)

    # 4️⃣ Reportes (usar gif como "learning curve")
    report_txt = save_step2_report(
        acc,
        losses,
        path_surface,
        gif_path,  # ahora usamos el GIF
        path_dashboard
    )

    report_html = save_step2_report_html(
        acc,
        losses,
        path_surface,
        gif_path,
        path_dashboard
    )

    should_show = (not args.no_show) and sys.platform.startswith("win")

    if should_show:
        open_file(gif_path); time.sleep(args.open_delay)
        open_file(path_surface); time.sleep(args.open_delay)
        open_file(path_dashboard); time.sleep(args.open_delay)
        open_file(report_txt); time.sleep(0.4)
        open_file(report_html); time.sleep(0.4)

if __name__ == "__main__":
    main()