# Predicción de precios de productos a partir de texto

Modelo de NLP + regresión que estima el precio de un producto de segunda mano a partir de su nombre, marca, categoría y descripción.

## Problema

En un marketplace, los vendedores no siempre saben qué precio ponerle a lo que venden. Un buen sugeridor de precios ayuda a vender más rápido y a precios justos. La dificultad está en que la mayor parte de la información está en **texto libre**.

## Datos

[Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge) (Kaggle): ~1.48 millones de productos publicados en la app Mercari.

- **Columnas:** nombre, estado del artículo, categoría, marca, envío, descripción y precio (objetivo)
- **Muestra usada:** 100,000 productos al azar (`random_state=42`), para entrenar rápido en una laptop

## Enfoque

1. **Limpieza:** se eliminan productos con precio 0, se rellenan marcas y categorías vacías y se quitan descripciones genéricas ("No description yet").
2. **Variable objetivo:** `log(1 + precio)`. Los precios están muy sesgados a la derecha, y el logaritmo evita que los productos caros dominen el error.
3. **Features:**
   - Texto combinado (nombre + marca + categoría + descripción) → **TF-IDF** con unigramas y bigramas, 50,000 términos
   - Estado del artículo y envío → **One-Hot Encoding**
4. **Evaluación:** 80% entrenamiento, 20% prueba. Métrica: **RMSLE** (menor es mejor).

## Resultados

| Modelo | RMSLE | Tiempo de entrenamiento |
|---|---|---|
| **Ridge** | **0.503** | 1.5 s |
| LightGBM | 0.504 | 87 s |
| LinearSVR | 0.525 | 0.9 s |
| Baseline (siempre el precio promedio) | 0.742 | — |

**Hallazgos:**
- Ridge reduce el error un **32%** frente al baseline y es el más rápido.
- Agregar estado y envío a las features de texto bajó el RMSLE de Ridge de 0.514 a 0.503.
- LightGBM queda prácticamente empatado, pero tarda ~58 veces más. Con miles de columnas dispersas de TF-IDF, un modelo lineal es la mejor relación entre precisión y costo.

## Estructura

```
├── notebooks/
│   ├── 01_exploracion.ipynb         # Análisis exploratorio
│   ├── 02_modelo_base.ipynb         # Limpieza, baseline y primer modelo
│   └── 03_comparacion_modelos.ipynb # Comparación de modelos
├── src/
│   ├── data.py       # Carga y limpieza
│   ├── features.py   # TF-IDF + One-Hot
│   └── train.py      # Entrenamiento, evaluación y guardado del modelo
├── metricas.json     # Resultados del último entrenamiento
└── requirements.txt
```

## Cómo correrlo

```bash
git clone https://github.com/joynova80-cell/regresion-precios-texto.git
cd regresion-precios-texto
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash). En Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

Descarga `train.tsv` desde [Kaggle](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data), ponlo en `data/` y entrena:

```bash
python -m src.train
```

## Próximos pasos

- Promediar las predicciones de Ridge y LightGBM (ensemble)
- Probar embeddings de oraciones en vez de TF-IDF
- Entrenar con el dataset completo
- Exponer el modelo como API con FastAPI y Docker

## Autor

**Joyland Adames**, estudiante de Inteligencia Artificial en ITLA · [GitHub](https://github.com/joynova80-cell)