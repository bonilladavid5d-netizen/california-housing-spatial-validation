"""Particiones: split original, validación aleatoria y validación espacial con margen."""
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.neighbors import BallTree


def particion_holdout(n, cfg):
    p = cfg["disenos"]["holdout_original"]
    idx = np.arange(n)
    train, test = train_test_split(idx, test_size=p["test_size"], random_state=p["random_state"])
    # Se conserva el orden de train_test_split: el bootstrap de Random Forest depende del orden de las filas,
    # y ordenar los índices impediría reproducir exactamente el notebook original.
    return [{"fold": 1, "train_antes": train, "train": train, "test": test}]


def particion_aleatoria(n, cfg):
    p = cfg["disenos"]["aleatorio_5f"]
    kf = KFold(n_splits=p["n_splits"], shuffle=p["shuffle"], random_state=p["random_state"])
    return [
        {"fold": k + 1, "train_antes": tr, "train": tr, "test": te}
        for k, (tr, te) in enumerate(kf.split(np.arange(n)))
    ]


def celdas(lat, lon, tamano):
    """Identificador de celda: floor(lat/tamano) y floor(lon/tamano)."""
    i = np.floor(np.asarray(lat) / tamano).astype(int)
    j = np.floor(np.asarray(lon) / tamano).astype(int)
    return i, j, np.array([f"{a}_{b}" for a, b in zip(i, j)])


def distancia_minima_km(origen, destino, radio_km):
    """Distancia haversine de cada punto de `origen` al punto más cercano de `destino` (grados lat/lon)."""
    arbol = BallTree(np.radians(destino), metric="haversine")
    d, _ = arbol.query(np.radians(origen), k=1)
    return d[:, 0] * radio_km


def particion_espacial(lat, lon, cfg):
    p = cfg["disenos"]["espacial_5f"]
    _, _, grupo = celdas(lat, lon, p["tamano_celda_grados"])
    coords = np.column_stack([lat, lon])
    gkf = GroupKFold(n_splits=p["n_splits"])
    folds = []
    for k, (tr, te) in enumerate(gkf.split(coords, groups=grupo)):
        dmin = distancia_minima_km(coords[tr], coords[te], p["radio_tierra_km"])
        conservar = dmin >= p["margen_km"]
        folds.append({
            "fold": k + 1,
            "train_antes": tr,
            "train": tr[conservar],
            "test": te,
            "excluidas": tr[~conservar],
            "dist_min_km_conservadas": float(dmin[conservar].min()) if conservar.any() else np.nan,
        })
    return folds, grupo
