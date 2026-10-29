"""Especificaciones fijas de los cuatro modelos (ver docs/PROTOCOLO.md)."""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

MODELOS = ["media", "mco", "rf", "gb"]
NOMBRES = {
    "media": "Media de entrenamiento",
    "mco": "MCO",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
}


def crear_modelo(nombre, cfg):
    m = cfg["modelos"]
    if nombre == "media":
        return DummyRegressor(strategy="mean")
    if nombre == "mco":
        return LinearRegression(fit_intercept=True)
    if nombre == "rf":
        # n_jobs solo paraleliza; con random_state fijo el resultado no depende de él.
        return RandomForestRegressor(**m["rf"], n_jobs=-1)
    if nombre == "gb":
        return GradientBoostingRegressor(**m["gb"])
    raise ValueError(nombre)


def ajustar_y_predecir(X, y, train, test, cfg):
    """Entrena los cuatro modelos con las mismas filas y predice las mismas filas.

    Devuelve (predicciones, estados): un fallo se registra en `estados` y deja NaN, sin cambiar el diseño.
    """
    preds, estados = {}, {}
    for nombre in MODELOS:
        try:
            if len(train) == 0:
                raise ValueError("entrenamiento vacío")
            modelo = crear_modelo(nombre, cfg).fit(X[train], y[train])
            preds[nombre] = modelo.predict(X[test])
            estados[nombre] = "ok"
        except Exception as e:  # se reporta y la verificación posterior detiene la ejecución
            preds[nombre] = np.full(len(test), np.nan)
            estados[nombre] = f"fallo: {e}"
    return preds, estados
