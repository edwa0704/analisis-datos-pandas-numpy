📊 Análisis de Datos con Pandas, NumPy & SciPy
Paso 1: Ingeniería de Características Avanzada
Este proyecto implementa un pipeline de transformación y análisis de datos utilizando herramientas avanzadas del ecosistema científico de Python.
Tecnologías utilizadas
Pandas → Merge complejo y window functions (rolling 48h)
NumPy → Target Encoding vectorizado (sin bucles for)
SciPy → Filtro Savitzky–Golay y test estadístico Kolmogorov–Smirnov
El objetivo es generar features avanzadas listas para modelado o análisis exploratorio.

📁 Estructura del Proyecto:
analisis-datos-pandas-numpy/
├── src/
│   └── pipeline_paso1.py
├── data/
├── reports/
├── requirements.txt
└── README.md

⚙️ Instalación:
1️⃣ Crear entorno virtual (python -m venv .venv)
2️⃣ Activar entorno virtual (.\.venv\Scripts\Activate.ps1)
3️⃣ Instalar dependencias (pip install -r requirements.txt)

🚀 Ejecutar Paso 1 
Ejecución básica (python src/pipeline_paso1.py)
Ejecución con parámetros personalizados (python src/pipeline_paso1.py --n_users 5000 --n_tx 200000 --seed 42)

Parámetros disponibles: 
--n_users → cantidad de usuarios
--n_tx → cantidad de transacciones
--seed → semilla para reproducibilidad

🧠 Funcionalidades Implementadas

✔ Generación de datos sintéticos
✔ Merge validado (many_to_one)
✔ Feature amount = price * volume
✔ Rolling 48h por usuario (avg_amount_48h)
✔ Target Encoding vectorizado (segment_te)
✔ Suavizado Savitzky–Golay (price_smooth)
✔ KS Test (normal vs sospechosa)
✔ Optimización de tipos
✔ Validaciones de calidad de datos
✔ Export de dataset y reportes

📤 Archivos Generados
Al ejecutar el pipeline se generan:

data/paso1_features.csv
reports/paso1_reporte.txt
reports/data_quality_summary.csv
reports/data_quality_nulls.csv

✅ Verificación 
La ejecución correcta debe mostrar en terminal:

Shapes de usuarios y transacciones
Confirmación de columnas generadas
Resultado del KS test (stat y p-value)
Validación de duplicados y nulos

📝 Historial de Desarrollo
Estructura base

Generación de datos

Merge y features

Rolling 48h

Target Encoding

Suavizado con SciPy

KS Test

Reportes y validaciones

Correcciones finales
🔜 Próximo Paso
Paso 2: Visualización de Alta Dimensionalidad
Reducción de dimensionalidad (t-SNE o UMAP)
Visualización con Seaborn
Representación de múltiples variables en 2D