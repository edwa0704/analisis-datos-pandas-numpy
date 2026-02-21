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

📊 Paso 2: Visualización de Alta Dimensionalidad (Matplotlib & Seaborn)
🎯 Objetivo

Visualizar datos de alta dimensionalidad (>3 dimensiones) mediante reducción a 2 componentes y modelado supervisado.

Se implementa:
- 🔹 Reducción de dimensionalidad con **t-SNE (Scikit-learn)**
- 🔹 Visualización con **Seaborn (color = price, tamaño = volume)**
- 🔹 Modelo de clasificación (Logistic Regression)
- 🔹 Superficie de decisión
- 🔹 Curva de aprendizaje (Loss vs Epochs)
- 🔹 Curva en tiempo real (GIF)
- 🔹 Dashboard final 2x2 (Matplotlib)
- 🔹 Exportación de métricas en CSV

📁 Estructura del Proyecto:
src/
 ├── pipeline_paso1.py
 ├── paso2_visualizacion.py
 └── paso2_modelo.py

📦 Requisitos

Python 3.10+

Instalar dependencias:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
▶ Ejecutar Paso 2:
python src/paso2_modelo.py

Ejecución personalizada:
python src/paso2_modelo.py --epochs 15 --sample 20000

Opciones disponibles:
Argumento	Descripción
--epochs	Número de épocas para SGDClassifier
--sample	Tamaño de muestra para entrenamiento
--no-show	No abrir imágenes automáticamente

📁 Archivos generados

Se guardan en la carpeta reports/:
paso2_decision_surface.png
paso2_learning_curve_realtime.gif
paso2_2x2_dashboard.png
paso2_reporte.txt
paso2_reporte.html
paso2_metrics.csv

📈 Descripción técnica
🔹 Reducción de dimensionalidad
Se utiliza TSNE de Scikit-learn para transformar múltiples variables numéricas en 2 componentes visualizables.

🔹 Visualización
Se utiliza Seaborn scatterplot donde:
Color representa el precio (price)
Tamaño del punto representa el volumen (volume)

🔹 Modelo
Se entrena:
Logistic Regression (modelo base)
SGDClassifier para curva de aprendizaje

🔹 Dashboard 2x2 incluye:
Scatter PCA 2D
Superficie de decisión
Curva Loss vs Epochs
Distribución de precios

🧪 Evidencia reproducible

El proyecto puede clonarse y ejecutarse desde cero:
git clone <URL_DEL_REPOSITORIO>
cd analisis-datos-pandas-numpy
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/paso2_modelo.py

✅ Resultado
El Paso 2 cumple con:
Visualización de alta dimensionalidad
Modelado supervisado
Subplots 2x2 requeridos
Curva de aprendizaje en tiempo real
Exportación de métricas

🧠 Flujo de ejecución del Paso 2
1. Se cargan las features generadas en Paso 1.
2. Se reduce dimensionalidad con t-SNE.
3. Se entrena modelo base (Logistic Regression).
4. Se genera superficie de decisión.
5. Se entrena SGDClassifier por épocas.
6. Se construye dashboard 2x2.
7. Se exportan métricas y reportes.