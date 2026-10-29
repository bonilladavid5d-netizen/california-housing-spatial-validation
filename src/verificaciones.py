"""Comprobaciones del protocolo. Cada una devuelve (nombre, ok, detalle); cualquier fallo detiene la ejecución."""
import numpy as np
import pandas as pd

from .metricas import metricas
from .modelos import MODELOS


def haversine_min_km(origen, destino, radio_km, bloque=2000):
    """Distancia mínima por fuerza bruta, independiente del BallTree usado en la partición."""
    lat2 = np.radians(destino[:, 0])[None, :]
    lon2 = np.radians(destino[:, 1])[None, :]
    salida = np.empty(len(origen))
    for i in range(0, len(origen), bloque):
        o = origen[i:i + bloque]
        lat1 = np.radians(o[:, 0])[:, None]
        lon1 = np.radians(o[:, 1])[:, None]
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        salida[i:i + bloque] = (2 * radio_km * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).min(axis=1)
    return salida


def verificar_particion(diseno, folds, n, cubre_todo):
    res = []
    for f in folds:
        inter = np.intersect1d(f["train"], f["test"])
        res.append((f"{diseno} f{f['fold']}: sin solapamiento entrenamiento-prueba", len(inter) == 0,
                    f"intersección={len(inter)}"))
        sub = np.setdiff1d(f["train"], f["train_antes"])
        res.append((f"{diseno} f{f['fold']}: entrenamiento final ⊆ entrenamiento inicial", len(sub) == 0,
                    f"filas fuera={len(sub)}"))
    tests = np.concatenate([f["test"] for f in folds])
    if cubre_todo:
        ok = len(tests) == n and len(np.unique(tests)) == n
        res.append((f"{diseno}: cada fila aparece en prueba exactamente una vez", ok,
                    f"filas en prueba={len(tests)}, únicas={len(np.unique(tests))}, n={n}"))
    return res


def verificar_espacial(folds, grupo, coords, cfg):
    p = cfg["disenos"]["espacial_5f"]
    res = []
    fold_de = np.empty(len(grupo), dtype=int)
    for f in folds:
        fold_de[f["test"]] = f["fold"]
    folds_por_celda = pd.Series(fold_de).groupby(grupo).nunique()
    res.append(("espacial: cada celda pertenece a un solo fold", bool((folds_por_celda == 1).all()),
                f"celdas={len(folds_por_celda)}, con más de un fold={(folds_por_celda > 1).sum()}"))
    for f in folds:
        tr, te = f["train"], f["test"]
        d_tr = haversine_min_km(coords[tr], coords[te], p["radio_tierra_km"])
        ok = bool(d_tr.min() >= p["margen_km"])
        res.append((f"espacial f{f['fold']}: distancia mínima entrenamiento-prueba ≥ {p['margen_km']} km", ok,
                    f"mínima={d_tr.min():.3f} km (fuerza bruta)"))
        if len(f["excluidas"]):
            d_ex = haversine_min_km(coords[f["excluidas"]], coords[te], p["radio_tierra_km"])
            ok = bool(d_ex.max() < p["margen_km"])
            res.append((f"espacial f{f['fold']}: toda fila excluida está a < {p['margen_km']} km de prueba", ok,
                        f"excluidas={len(f['excluidas'])}, máxima={d_ex.max():.3f} km"))
        celdas_tr = set(grupo[tr])
        celdas_te = set(grupo[te])
        res.append((f"espacial f{f['fold']}: ninguna celda de prueba en entrenamiento",
                    len(celdas_tr & celdas_te) == 0, f"celdas compartidas={len(celdas_tr & celdas_te)}"))
    return res


def verificar_predicciones(pred, n, disenos_cv):
    res = []
    cols = [f"pred_{m}" for m in MODELOS]
    nulos = int(pred[cols].isna().sum().sum())
    res.append(("todas las predicciones existen (mismas filas para los cuatro modelos)", nulos == 0,
                f"predicciones faltantes={nulos}"))
    for d in disenos_cv:
        sub = pred[pred["diseno"] == d]
        conteo = sub["row_id"].value_counts()
        ok = len(sub) == n and sub["row_id"].nunique() == n and bool((conteo == 1).all())
        res.append((f"{d}: una predicción fuera de muestra por fila y modelo", ok,
                    f"filas={len(sub)}, únicas={sub['row_id'].nunique()}"))
    return res


def recalcular_metricas(pred, met, tol):
    """Recalcula métricas por fold y globales desde predicciones leídas de disco y compara con la tabla."""
    res = []
    max_dif = 0.0
    for _, fila in met.iterrows():
        sub = pred[pred["diseno"] == fila["diseno"]]
        if fila["nivel"] == "fold":
            sub = sub[sub["fold"] == fila["fold"]]
        m = metricas(sub["observado"], sub[f"pred_{fila['modelo']}"])
        for k in ("rmse", "mae", "r2"):
            max_dif = max(max_dif, abs(m[k] - fila[k]))
        if m["n"] != fila["n_prueba"]:
            max_dif = np.inf
    res.append(("métricas recalculadas desde outputs/predicciones.csv coinciden", max_dif <= tol,
                f"diferencia máxima={max_dif:.2e} (tolerancia {tol:.0e}), filas de métricas={len(met)}"))
    glob = met[met["nivel"] == "global"]
    fold = met[met["nivel"] == "fold"]
    prom = fold.groupby(["diseno", "modelo"])["rmse"].mean()
    dif = (glob.set_index(["diseno", "modelo"])["rmse"] - prom).abs().dropna()
    res.append(("informativa: RMSE global (agrupado) difiere del promedio de RMSE por fold", True,
                f"diferencia máxima={dif.max():.4f}; la tabla usa siempre el agrupado"))
    return res
