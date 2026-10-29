"""Métricas de error. Las globales se calculan sobre predicciones agrupadas, nunca promediando folds."""
import numpy as np


def metricas(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    e = yhat - y
    sse = float(np.sum(e ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(sse / len(y))),
        "mae": float(np.mean(np.abs(e))),
        "r2": 1 - sse / sst if sst > 0 else np.nan,
        "sesgo_medio": float(np.mean(e)),
    }


def a_usd(valor, cfg):
    return valor * cfg["datos"]["unidad_objetivo_usd"]
