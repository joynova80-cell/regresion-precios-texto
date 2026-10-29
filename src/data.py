import pandas as pd

DATA_PATH = "data/train.tsv"


def cargar_datos(ruta=DATA_PATH, n_muestras=100_000, semilla=42):
    """Carga el dataset de Mercari y toma una muestra aleatoria."""
    df = pd.read_csv(ruta, sep="\t")
    if n_muestras:
        df = df.sample(n_muestras, random_state=semilla).reset_index(drop=True)
    return df


def limpiar(df):
    """Quita precios en 0, rellena vacíos y crea la columna de texto combinado."""
    df = df[df["price"] > 0].copy()
    df["brand_name"] = df["brand_name"].fillna("sin_marca")
    df["category_name"] = df["category_name"].fillna("sin_categoria")
    df["item_description"] = df["item_description"].fillna("").replace("No description yet", "")
    df["texto"] = (
        df["name"] + " " + df["brand_name"] + " "
        + df["category_name"] + " " + df["item_description"]
    )
    return df