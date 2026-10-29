"""Carga, huellas y auditoría de la base California Housing (censo 1990)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# Definiciones según la documentación de scikit-learn (fetch_california_housing).
DICCIONARIO = [
    ("MedInc", "Ingreso mediano de los hogares del block group", "mediana", "decenas de miles de USD de 1990"),
    ("HouseAge", "Edad mediana de las viviendas", "mediana", "años"),
    ("AveRooms", "Habitaciones por hogar", "promedio", "habitaciones"),
    ("AveBedrms", "Dormitorios por hogar", "promedio", "dormitorios"),
    ("Population", "Población del block group", "total", "personas"),
    ("AveOccup", "Personas por hogar", "promedio", "personas"),
    ("Latitude", "Latitud del block group", "coordenada", "grados"),
    ("Longitude", "Longitud del block group", "coordenada", "grados"),
    ("MedHouseVal", "Valor mediano de la vivienda (objetivo)", "mediana", "cientos de miles de USD de 1990"),
]


class ErrorIntegridad(RuntimeError):
    """Fallo de una comprobación que obliga a detener la ejecución."""


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def verificar_originales(cfg, base):
    """Compara la huella de cada archivo original con la registrada en config.

    Ausente y opcional: estado NO DISPONIBLE (N/A) (no cuenta como superada). Presente con otra huella: detiene la ejecución.
    """
    filas = []
    for rel, esperado in cfg["originales"]["archivos"].items():
        ruta = (base / rel).resolve()
        if not ruta.exists():
            if cfg["originales"]["exigir_presencia"]:
                raise ErrorIntegridad(f"No se encontró el original {rel}")
            filas.append({"archivo": rel, "presente": False, "sha256": None, "estado": "NO DISPONIBLE"})
            continue
        obtenido = sha256(ruta)
        if obtenido != esperado:
            raise ErrorIntegridad(f"La huella de {rel} cambió: {obtenido} != {esperado}")
        filas.append({"archivo": rel, "presente": True, "sha256": obtenido, "estado": "PASS"})
    return filas


def cargar_datos(cfg, base):
    ruta = base / cfg["datos"]["ruta"]
    huella = sha256(ruta)
    if huella != cfg["datos"]["sha256"]:
        raise ErrorIntegridad(f"La huella del CSV de trabajo no coincide: {huella}")
    df = pd.read_csv(ruta)
    columnas = cfg["datos"]["predictores"] + [cfg["datos"]["objetivo"]]
    if list(df.columns) != columnas:
        raise ErrorIntegridad(f"Columnas inesperadas: {list(df.columns)}")
    if len(df) != cfg["datos"]["filas_esperadas"]:
        raise ErrorIntegridad(f"Se esperaban {cfg['datos']['filas_esperadas']} filas y hay {len(df)}")
    df.index.name = "row_id"
    return df


def auditar(df, cfg):
    """Resumen de calidad de datos. Lanza ErrorIntegridad si no se cumplen los supuestos del protocolo."""
    y = cfg["datos"]["objetivo"]
    techo = cfg["datos"]["techo_objetivo"]
    coords = df[["Latitude", "Longitude"]]
    tam_coord = coords.groupby(["Latitude", "Longitude"]).size()
    n_techo = int(np.isclose(df[y], techo, rtol=0, atol=1e-9).sum())

    resumen = {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "faltantes_por_columna": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicados_completos": int(df.duplicated().sum()),
        "pares_coordenadas_unicos": int(len(tam_coord)),
        "filas_con_coordenada_repetida": int(tam_coord[tam_coord > 1].sum()),
        "max_filas_misma_coordenada": int(tam_coord.max()),
        "objetivo_maximo": float(df[y].max()),
        "objetivo_minimo": float(df[y].min()),
        "n_en_techo_objetivo": n_techo,
        "pct_en_techo_objetivo": round(100 * n_techo / len(df), 3),
        "n_HouseAge_en_maximo_52": int((df["HouseAge"] == df["HouseAge"].max()).sum()),
        "n_MedInc_en_maximo": int((df["MedInc"] == df["MedInc"].max()).sum()),
        "MedInc_maximo": float(df["MedInc"].max()),
        "n_AveOccup_mayor_10": int((df["AveOccup"] > 10).sum()),
        "n_AveRooms_mayor_20": int((df["AveRooms"] > 20).sum()),
    }
    if sum(resumen["faltantes_por_columna"].values()) != 0:
        raise ErrorIntegridad("Hay valores faltantes")
    if n_techo != cfg["datos"]["n_techo_esperado"]:
        raise ErrorIntegridad(f"Se esperaban {cfg['datos']['n_techo_esperado']} filas en el techo y hay {n_techo}")

    dic = pd.DataFrame(DICCIONARIO, columns=["variable", "definicion", "agregacion", "unidad"])
    desc = df.describe().T.reset_index().rename(columns={"index": "variable"})
    dic = dic.merge(desc, on="variable")
    return resumen, dic
