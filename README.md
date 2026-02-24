📊 Análisis de Datos con Pandas, NumPy & SciPy

Proyecto avanzado de ingeniería de características, modelado supervisado y visualización científica utilizando el ecosistema científico de Python.

🚀 Funcionalidades

✔ Generación de datos sintéticos
✔ Feature Engineering avanzado
✔ Validaciones estadísticas (SciPy)
✔ Modelado supervisado (scikit-learn)
✔ Reducción de dimensionalidad (PCA / t-SNE)
✔ Dashboard 2x2 profesional
✔ GIF de curva de aprendizaje en tiempo real
✔ Exportación automática de reportes (CSV / PNG / GIF / HTML / TXT)
✔ Validación automática del entorno (check_env.py)

📦 Dependencias del Proyecto

El proyecto está dividido en dos niveles para evitar instalar librerías innecesarias.

🔹 1️⃣ requirements-core.txt (Ejecución mínima)

Contiene únicamente lo necesario para ejecutar el proyecto:

numpy

pandas

scipy

scikit-learn

matplotlib

imageio

imageio-ffmpeg

tqdm

👉 Usa este archivo si solo quieres ejecutar el proyecto.

Instalación:

python -m pip install -r requirements-core.txt
🔹 2️⃣ requirements-dev.txt (Desarrollo)

Incluye herramientas adicionales para desarrollo:

jupyter

ipykernel

pytest

black

ruff

👉 Instálalo solo si vas a trabajar en notebooks, testing o desarrollo interno.

Instalación:

python -m pip install -r requirements-dev.txt

Requisitos previos:
- Python 3.9 o superior
- Git instalado

🚀 Quick Start

Clona el repositorio y crea el entorno virtual:

git clone https://github.com/edwa0704/analisis-datos-pandas-numpy.git
cd analisis-datos-pandas-numpy
python -m venv .venv
🖥 Activación según tu Terminal

Después de crear .venv, activa el entorno según tu sistema:

🟦 CMD (Símbolo del sistema)
.\.venv\Scripts\activate.bat
python -m pip install -r requirements-core.txt

🟨 PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt

Si PowerShell bloquea la activación:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Luego vuelve a intentar activar.

🟩 Git Bash
source .venv/Scripts/activate
python -m pip install -r requirements-core.txt

🟪 Linux / macOS
source .venv/bin/activate
python -m pip install -r requirements-core.txt
🔧 Método Alternativo Seguro (Si la Activación Falla)

Si el entorno virtual no activa correctamente, ejecuta directamente el Python del entorno:

Windows
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe check_env.py
Linux / macOS
./.venv/bin/python -m pip install -r requirements-core.txt
./.venv/bin/python check_env.py
🔍 Verificación del Entorno (Smoke Test)

Antes de ejecutar el proyecto, verifica que todo esté correcto:

python check_env.py

Este script:

✔ Verifica versión de Python
✔ Confirma librerías críticas instaladas
✔ Crea automáticamente las carpetas data/ y reports/ si no existen

Si todo aparece con ✅, el entorno está listo.

▶ Ejecución del Proyecto

Paso 1 – Ingeniería de características:

python src/pipeline_paso1.py

Paso 2 – Modelo supervisado:

python src/paso2_modelo.py --no-show

Visualización opcional:

python src/paso2_visualizacion.py --sample 8000
🎞 Dependencias Críticas

El proyecto genera GIF usando:

imageio

imageio-ffmpeg

Si aparece error relacionado con ffmpeg:

python -m pip install imageio imageio-ffmpeg

Siempre usar:

python -m pip install ...
PowerShell bloquea scripts

Ejecutar:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Creación automática de data/ y reports/

Después de ejecutar correctamente:

data/paso1_features.csv
reports/paso1_reporte.txt
reports/data_quality_summary.csv
reports/data_quality_nulls.csv
reports/paso2_decision_surface.png
reports/paso2_learning_curve_realtime.gif
reports/paso2_2x2_dashboard.png
reports/paso2_reporte.txt
reports/paso2_reporte.html
reports/paso2_metrics.csv
🧠 Dependencias críticas

⚙ Soporte opcional: PyTorch (CPU / GPU)

Si deseas experimentar con modelos adicionales:

CPU
pip install torch torchvision torchaudio
GPU (CUDA)

Visitar:
https://pytorch.org/get-started/locally/

Seleccionar:

Windows

pip

CUDA version correspondiente

🚨 Errores comunes y soluciones
❌ python no reconocido

Instalar Python marcando "Add Python to PATH".

❌ .venv no activa

Usar directamente:

.venv\Scripts\python.exe
❌ pip instala en otro Python

Siempre usar:

python -m pip install ...
❌ PowerShell bloquea scripts

Ejecutar:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
🎯 Buenas prácticas aplicadas

✔ Separación de dependencias (core / dev)
✔ Instalación mínima limpia
✔ Validación automática del entorno
✔ Creación automática de carpetas necesarias
✔ Documentación diferenciada por terminal
✔ Método alternativo robusto de ejecución

👨‍💻 Autor

Frank Edwar Pérez Bustillos
Ingeniería de Programación, IA y Software