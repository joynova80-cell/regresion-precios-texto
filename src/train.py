import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.data import cargar_datos, limpiar
from src.features import ConstructorFeatures

MODELS_DIR = Path("models")


def rmsle(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    print("Cargando datos...")
    df = limpiar(cargar_datos())

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    y_train = np.log1p(train_df["price"])
    y_test = np.log1p(test_df["price"])

    print("Construyendo features...")
    features = ConstructorFeatures()
    X_train = features.fit_transform(train_df)
    X_test = features.transform(test_df)

    print("Entrenando Ridge...")
    inicio = time.time()
    modelo = Ridge(alpha=1.0)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    metricas = {
        "modelo": "Ridge",
        "rmsle": round(rmsle(y_test, pred), 4),
        "rmsle_baseline": round(rmsle(y_test, np.full(len(y_test), y_train.mean())), 4),
        "segundos_entrenamiento": round(time.time() - inicio, 1),
    }
    print(metricas)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"features": features, "modelo": modelo}, MODELS_DIR / "ridge.joblib")
    with open("metricas.json", "w") as f:
        json.dump(metricas, f, indent=2)
    print("Modelo guardado en models/ridge.joblib")


if __name__ == "__main__":
    main()