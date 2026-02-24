import sys
import os
import importlib
from pathlib import Path

print("=" * 50)
print("🔍 CHECK ENVIRONMENT - SMOKE TEST")
print("=" * 50)

# -----------------------------
# 1️⃣ Verificar versión de Python
# -----------------------------
print("\n📌 Python Version:")
print(sys.version)

if sys.version_info < (3, 9):
    print("⚠️ Se recomienda Python 3.9 o superior.")
else:
    print("✅ Versión de Python compatible.")

# -----------------------------
# 2️⃣ Librerías críticas
# -----------------------------
required_packages = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "imageio",
    "tqdm",
]

print("\n📦 Verificando librerías críticas...\n")

missing = []

for package in required_packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"✅ {package} (version: {version})")
    except ImportError:
        print(f"❌ {package} NO instalado")
        missing.append(package)

if missing:
    print("\n🚨 Faltan las siguientes librerías:")
    for pkg in missing:
        print(f"- {pkg}")
    print("\nEjecuta:")
    print("python -m pip install -r requirements-core.txt")
else:
    print("\n🎉 Todas las librerías críticas están instaladas.")

# -----------------------------
# 3️⃣ Crear carpetas necesarias
# -----------------------------
print("\n📁 Verificando carpetas necesarias...")

folders = ["data", "reports"]

for folder in folders:
    path = Path(folder)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"📂 Carpeta creada: {folder}/")
    else:
        print(f"✅ Carpeta existente: {folder}/")

print("\n✔ CHECK COMPLETADO CORRECTAMENTE.")
print("=" * 50)