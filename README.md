📊 Análisis de Datos con Pandas, NumPy & SciPy

Proyecto de ingeniería de características y visualización avanzada utilizando el ecosistema científico de Python.

Este repositorio implementa:
✔ Generación de datos sintéticos
✔ Feature Engineering avanzado
✔ Validaciones estadísticas
✔ Modelado supervisado
✔ Visualización de alta dimensionalidad
✔ Dashboard 2x2 profesional
✔ Exportación automática de reportes

🧠 ¿Qué hace este proyecto?

Este proyecto simula un entorno real de análisis de transacciones:

Genera usuarios y transacciones.
Construye variables avanzadas (features).
Aplica técnicas estadísticas.
Reduce dimensionalidad.
Entrena modelos de clasificación.
Genera visualizaciones y reportes automáticos.
Está dividido en:

🔹 Paso 1 → Ingeniería de Características

🔹 Paso 2 → Modelado y Visualización

📁 Estructura del Proyecto
analisis-datos-pandas-numpy/
│
├── src/
│   ├── pipeline_paso1.py
│   ├── paso2_modelo.py
│   └── paso2_visualizacion.py
│
├── data/
├── reports/
├── requirements.txt
└── README.md
🧩 Requisitos Previos

Antes de empezar necesitas:

Python 3.10 o superior
Git instalado
Terminal (CMD o PowerShell)
Verificar Python:
python --version
📥 Cómo Clonar el Proyecto

1️⃣ Abrir la terminal
2️⃣ Ejecutar:

git clone https://github.com/edwa0704/analisis-datos-pandas-numpy.git
cd analisis-datos-pandas-numpy
⚙️ Instalación Paso a Paso

1️⃣ Crear entorno virtual
python -m venv .venv

2️⃣ Activar entorno virtual
CMD:
.\.venv\Scripts\activate.bat

PowerShell:
.\.venv\Scripts\Activate.ps1

Si está activo verás algo así:
(.venv)

3️⃣ Instalar dependencias
pip install -r requirements.txt

▶ Ejecutar Paso 1 – Ingeniería de Características

Ejecución básica:

python src/pipeline_paso1.py

Ejecución personalizada:
python src/pipeline_paso1.py --n_users 5000 --n_tx 200000 --seed 42
🔧 Parámetros disponibles

--n_users → Número de usuarios
--n_tx → Número de transacciones
--seed → Semilla para reproducibilidad

📤 Archivos Generados en Paso 1

Se crean automáticamente:

data/paso1_features.csv
reports/paso1_reporte.txt
reports/data_quality_summary.csv
reports/data_quality_nulls.csv

▶ Ejecutar Paso 2 
– Modelado y Visualización
python src/paso2_modelo.py

Ejecución personalizada:
python src/paso2_modelo.py --epochs 15 --sample 20000

🔧 Parámetros disponibles
--epochs → Número de épocas
--sample → Tamaño de muestra
--no-show → No abrir imágenes automáticamente

📤 Archivos Generados en Paso 2

En carpeta reports/:
paso2_decision_surface.png
paso2_learning_curve_realtime.gif
paso2_2x2_dashboard.png
paso2_reporte.txt
paso2_reporte.html
paso2_metrics.csv

🧪 ¿Cómo Saber si Funcionó Correctamente?

Paso 1 debe mostrar en terminal:
Shapes de usuarios y transacciones
Confirmación de columnas generadas
Resultado del KS test
Validación de nulos y duplicados
Paso 2 debe:
Mostrar scatter plot
Generar GIF en tiempo real
Crear dashboard 2x2
Exportar métricas

Si eso ocurre → ejecución correcta ✅

🚨 Errores Comunes y Soluciones
❌ Error: No se reconoce python
Instalar Python desde:
https://www.python.org/downloads/

❌ Error: No se reconoce git
Instalar Git desde:
https://git-scm.com/

❌ Error al activar entorno virtual
Asegurarse de usar el comando correcto según terminal (CMD o PowerShell).

❌ Error instalando SciPy
Usar Python 3.10 o 3.11 (algunas versiones no son compatibles con 3.13).

❌ No se generan imágenes
Eliminar carpeta reports/ y ejecutar nuevamente.

📊 Tecnologías Utilizadas

Pandas
NumPy
SciPy
Scikit-learn
Matplotlib
Seaborn

🔄 Flujo Completo del Proyecto

Generación de datos
Ingeniería de características
Validación estadística
Reducción de dimensionalidad
Entrenamiento de modelo
Curva de aprendizaje
Dashboard 2x2
Exportación de reportes

**Notas de mejora**

- Documentación: corregir y simplificar los comandos de activación y distinguir claramente CMD / PowerShell / Git Bash.
- Dependencias: separar un `requirements-core.txt` (mínimo) y un `requirements-dev.txt` (Jupyter, tests, tooling).
- Instalación: indicar dependencias críticas (`imageio`, `imageio-ffmpeg`) y cómo instalar `torch` para CPU/GPU.
- Robustez: asegurar que los scripts creen `data/` y `reports/` si no existen; añadir un `check_env.py` (smoke-test).
- Errores comunes: listar soluciones rápidas (activar .venv correctamente, usar `.venv/Scripts/python.exe` si hay problemas de shell).

Nota: 13