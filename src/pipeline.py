"""Ejecución completa del caso. Uso: python3 run.py (desde la carpeta del proyecto).

Cualquier comprobación fallida lanza ErrorIntegridad y termina con código distinto de cero.
"""
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm
from sklearn.inspection import permutation_importance

from . import figuras
from .datos import ErrorIntegridad, auditar, cargar_datos, sha256, verificar_originales
from .metricas import a_usd, metricas
from .modelos import MODELOS, ajustar_y_predecir, crear_modelo
from .particiones import celdas, particion_aleatoria, particion_espacial, particion_holdout
from .verificaciones import (recalcular_metricas, verificar_espacial, verificar_particion,
                             verificar_predicciones)

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
FIG = BASE / "figures"
DISENOS_CV = ["aleatorio_5f", "espacial_5f"]
NA = "NO DISPONIBLE"
log = logging.getLogger("california")


def configurar_log():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    # Se borran salidas previas: un fallo nunca deja archivos de una ejecución anterior como si fueran actuales.
    for viejo in [*OUT.glob("*.csv"), *OUT.glob("*.json"), *FIG.glob("*.png")]:
        viejo.unlink()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT / "ejecucion.log", mode="w", encoding="utf-8")])


def evaluar_diseno(diseno, folds, X, y, df, cfg):
    filas_pred, filas_met = [], []
    for f in folds:
        t0 = time.time()
        preds, estados = ajustar_y_predecir(X, y, f["train"], f["test"], cfg)
        te = f["test"]
        bloque = pd.DataFrame({
            "row_id": te, "diseno": diseno, "fold": f["fold"],
            "latitud": df["Latitude"].values[te], "longitud": df["Longitude"].values[te],
            "observado": y[te],
        })
        for m in MODELOS:
            bloque[f"pred_{m}"] = preds[m]
            mt = metricas(y[te], preds[m]) if estados[m] == "ok" else {"n": len(te)}
            filas_met.append({"diseno": diseno, "modelo": m, "nivel": "fold", "fold": f["fold"],
                              "n_entrenamiento": len(f["train"]), "n_prueba": len(te),
                              **{k: mt.get(k, np.nan) for k in ("rmse", "mae", "r2", "sesgo_medio")},
                              "estado": estados[m]})
        filas_pred.append(bloque)
        log.info(f"  {diseno} fold {f['fold']}: entrenamiento={len(f['train'])}, prueba={len(te)}, "
                 f"RMSE rf={filas_met[-2]['rmse']:.4f} ({time.time() - t0:.0f}s)")
        fallos = {m: e for m, e in estados.items() if e != "ok"}
        if fallos:
            log.error(f"  Fallo de estimación en {diseno} fold {f['fold']}: {fallos}")
    pred = pd.concat(filas_pred, ignore_index=True).sort_values("row_id", kind="stable")
    for m in MODELOS:
        mt = metricas(pred["observado"], pred[f"pred_{m}"])
        filas_met.append({"diseno": diseno, "modelo": m, "nivel": "global", "fold": np.nan,
                          "n_entrenamiento": np.nan, "n_prueba": mt["n"],
                          **{k: mt[k] for k in ("rmse", "mae", "r2", "sesgo_medio")},
                          "estado": "ok" if pred[f"pred_{m}"].notna().all() else "incompleto"})
    return pred, pd.DataFrame(filas_met)


def tabla_segmentos(pred, cfg):
    techo = cfg["datos"]["techo_objetivo"]
    bandas = cfg["segmentos"]["bandas_valor_observado"]
    etiquetas = [f"[{a:g}, {b:g})" for a, b in zip(bandas[:-1], bandas[1:])]
    en_techo = np.isclose(pred["observado"], techo, rtol=0, atol=1e-9)
    banda = pd.cut(pred["observado"], bins=bandas, right=False, labels=etiquetas).astype(object)
    banda[en_techo] = "techo (5,00001)"
    if banda.isna().any():
        raise ErrorIntegridad("Hay observaciones sin banda de valor asignada")
    pred = pred.assign(banda=banda.values, en_techo=np.where(en_techo, "en techo", "bajo techo"))
    filas = []
    for (d, tipo, seg), sub in [((d, "todas", "todas"), s) for d, s in pred.groupby("diseno")] + \
            [((d, "techo", s_), s) for (d, s_), s in pred.groupby(["diseno", "en_techo"])] + \
            [((d, "banda_valor", s_), s) for (d, s_), s in pred.groupby(["diseno", "banda"])]:
        for m in MODELOS:
            mt = metricas(sub["observado"], sub[f"pred_{m}"])
            filas.append({"diseno": d, "modelo": m, "tipo_segmento": tipo, "segmento": seg,
                          "n": mt["n"], "media_observado": sub["observado"].mean(),
                          "rmse": mt["rmse"], "mae": mt["mae"], "sesgo_medio": mt["sesgo_medio"],
                          "rmse_usd": a_usd(mt["rmse"], cfg), "mae_usd": a_usd(mt["mae"], cfg),
                          "sesgo_medio_usd": a_usd(mt["sesgo_medio"], cfg)})
    return pd.DataFrame(filas)


def tabla_celdas(pred, cfg):
    tam = cfg["segmentos"]["celda_mapa_grados"]
    i, j, cid = celdas(pred["latitud"], pred["longitud"], tam)
    pred = pred.assign(celda_id=cid, lat_centro=(i + 0.5) * tam, lon_centro=(j + 0.5) * tam)
    filas = []
    for (d, c), sub in pred.groupby(["diseno", "celda_id"]):
        fila = {"diseno": d, "celda_id": c, "lat_centro": sub["lat_centro"].iloc[0],
                "lon_centro": sub["lon_centro"].iloc[0], "n": len(sub),
                "media_observado": sub["observado"].mean()}
        for m in MODELOS:
            e = sub[f"pred_{m}"] - sub["observado"]
            fila[f"rmse_{m}"] = float(np.sqrt(np.mean(e ** 2)))
            fila[f"sesgo_{m}"] = float(e.mean())
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_distribucion(pred, cfg):
    filas = []
    for d, sub in pred.groupby("diseno"):
        for m in MODELOS:
            ae = a_usd((sub[f"pred_{m}"] - sub["observado"]).abs(), cfg)
            filas.append({"diseno": d, "modelo": m, "n": len(sub),
                          **{f"p{q}_error_abs_usd": float(np.percentile(ae, q)) for q in (50, 75, 90, 95, 99)},
                          **{f"pct_error_abs_menor_{k // 1000}k_usd": float(100 * (ae < k).mean())
                             for k in (25000, 50000, 100000)}})
    return pd.DataFrame(filas)


def interpretacion(X, y, nombres, fold, cfg):
    """Coeficientes MCO (errores HC3) e importancias, sobre el split original. Asociaciones, no efectos."""
    tr, te = fold["train"], fold["test"]
    ols = sm.OLS(y[tr], sm.add_constant(pd.DataFrame(X[tr], columns=nombres))).fit(cov_type="HC3")
    ic = ols.conf_int()
    coef = pd.DataFrame({"variable": ols.params.index, "coeficiente": ols.params.values,
                         "error_estandar_hc3": ols.bse.values, "ic95_inf": ic[0].values,
                         "ic95_sup": ic[1].values, "p_valor": ols.pvalues.values})
    filas = []
    for m in ("rf", "gb"):
        modelo = crear_modelo(m, cfg).fit(X[tr], y[tr])
        perm = permutation_importance(modelo, X[te], y[te], n_repeats=10, random_state=cfg["semilla"],
                                      scoring="neg_root_mean_squared_error")
        for k, v in enumerate(nombres):
            filas.append({"modelo": m, "variable": v,
                          "importancia_impureza": float(modelo.feature_importances_[k]),
                          "aumento_rmse_permutacion": float(perm.importances_mean[k]),
                          "aumento_rmse_permutacion_de": float(perm.importances_std[k])})
    return coef, pd.DataFrame(filas)


def reproduccion_original(met_hold, X, y, cfg):
    ref = cfg["referencias_revision_independiente"]
    g = met_hold.set_index("modelo")
    mco_full = crear_modelo("mco", cfg).fit(X, y)
    m_in = metricas(y, mco_full.predict(X))
    filas = [
        ("MCO dentro de muestra (base completa, como en el notebook)", m_in["rmse"], m_in["r2"], 0.724100, 0.606233, np.nan, np.nan),
        ("Media de entrenamiento, prueba 20%", g.loc["media", "rmse"], g.loc["media", "r2"], np.nan, np.nan, ref["media_rmse"], np.nan),
        ("MCO, prueba 20%", g.loc["mco", "rmse"], g.loc["mco", "r2"], np.nan, np.nan, ref["mco_rmse"], ref["mco_r2"]),
        ("Random Forest, prueba 20%", g.loc["rf", "rmse"], g.loc["rf", "r2"], 0.496417, 0.814688, ref["rf_rmse"], ref["rf_r2"]),
        ("Gradient Boosting, prueba 20%", g.loc["gb", "rmse"], g.loc["gb", "r2"], 0.504268, 0.808781, ref["gb_rmse"], ref["gb_r2"]),
    ]
    t = pd.DataFrame(filas, columns=["evaluacion", "rmse", "r2", "rmse_notebook", "r2_notebook",
                                     "rmse_referencia", "r2_referencia"])
    t["dif_rmse_vs_referencia"] = t["rmse"] - t["rmse_referencia"]
    t["dif_r2_vs_referencia"] = t["r2"] - t["r2_referencia"]
    coincide = (t["dif_rmse_vs_referencia"].abs() < 5e-7) & (t["dif_r2_vs_referencia"].abs().fillna(0) < 5e-7)
    t["coincide_referencia_6_decimales"] = np.where(t["rmse_referencia"].isna(), "sin referencia",
                                                    np.where(coincide, "sí", "no"))
    return t


def main():
    configurar_log()
    t_inicio = time.time()
    cfg = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    checks = []

    log.info("1. Integridad de originales y datos")
    originales_antes = verificar_originales(cfg, BASE)
    df = cargar_datos(cfg, BASE)
    auditoria, diccionario = auditar(df, cfg)
    checks += [("huella SHA-256 del CSV de trabajo", True, cfg["datos"]["sha256"][:16] + "…"),
               ("estructura: 20.640 filas, 9 columnas, sin faltantes", True, f"{auditoria['filas']}×{auditoria['columnas']}"),
               ("965 observaciones en el techo 5,00001", True, str(auditoria["n_en_techo_objetivo"]))]
    for o in originales_antes:
        checks.append((f"original opcional {o['archivo']} sin cambios (antes)", _estado_original(o),
                       "huella coincide" if o["presente"] else "NO DISPONIBLE en esta copia; no se verifica"))
    auditoria["huellas"] = {"csv_trabajo": sha256(BASE / cfg["datos"]["ruta"]),
                            **{o["archivo"]: o["sha256"] for o in originales_antes}}
    diccionario.to_csv(OUT / "diccionario_y_descriptivas.csv", index=False)

    nombres = cfg["datos"]["predictores"]
    X = df[nombres].to_numpy()
    y = df[cfg["datos"]["objetivo"]].to_numpy()
    n = len(df)
    lat, lon = df["Latitude"].to_numpy(), df["Longitude"].to_numpy()

    log.info("2. Particiones")
    p_hold = particion_holdout(n, cfg)
    p_ale = particion_aleatoria(n, cfg)
    p_esp, grupo = particion_espacial(lat, lon, cfg)
    checks += verificar_particion("holdout_original", p_hold, n, cubre_todo=False)
    checks += verificar_particion("aleatorio_5f", p_ale, n, cubre_todo=True)
    checks += verificar_particion("espacial_5f", p_esp, n, cubre_todo=True)
    checks += verificar_espacial(p_esp, grupo, np.column_stack([lat, lon]), cfg)
    detener_si_falla(checks, "particiones")

    ci, cj, _ = celdas(lat, lon, cfg["disenos"]["espacial_5f"]["tamano_celda_grados"])
    asign = pd.DataFrame({"row_id": np.arange(n), "latitud": lat, "longitud": lon,
                          "celda_lat": ci, "celda_lon": cj, "celda_id": grupo})
    asign["holdout_original"] = "entrenamiento"
    asign.loc[p_hold[0]["test"], "holdout_original"] = "prueba"
    for nombre, folds in (("fold_aleatorio", p_ale), ("fold_espacial", p_esp)):
        for f in folds:
            asign.loc[f["test"], nombre] = f["fold"]
        asign[nombre] = asign[nombre].astype(int)
    excl = {i: [] for i in range(n)}
    for f in p_esp:
        for i in f["excluidas"]:
            excl[i].append(str(f["fold"]))
    asign["excluida_por_margen_en_folds"] = [";".join(excl[i]) for i in range(n)]
    asign.to_csv(OUT / "asignacion_folds.csv", index=False)

    tam = []
    for d, folds in (("holdout_original", p_hold), ("aleatorio_5f", p_ale), ("espacial_5f", p_esp)):
        for f in folds:
            antes, despues = len(f["train_antes"]), len(f["train"])
            tam.append({"diseno": d, "fold": f["fold"], "n_prueba": len(f["test"]),
                        "n_celdas_prueba": len(set(grupo[f["test"]])),
                        "n_entrenamiento_antes_margen": antes, "n_excluidas_margen": antes - despues,
                        "n_entrenamiento_final": despues, "pct_excluido": round(100 * (antes - despues) / antes, 2),
                        "dist_min_entrenamiento_prueba_km": f.get("dist_min_km_conservadas", np.nan)})
    tam = pd.DataFrame(tam)
    tam.to_csv(OUT / "tamanos_entrenamiento.csv", index=False)
    log.info("\n" + tam.to_string(index=False))

    log.info("3. Estimación y predicción fuera de muestra")
    preds, mets = [], []
    for d, folds in (("holdout_original", p_hold), ("aleatorio_5f", p_ale), ("espacial_5f", p_esp)):
        p, m = evaluar_diseno(d, folds, X, y, df, cfg)
        preds.append(p)
        mets.append(m)
    pred = pd.concat(preds, ignore_index=True)
    met = pd.concat(mets, ignore_index=True)
    fallos = met[met["estado"] != "ok"]
    checks.append(("todos los modelos se estimaron en todos los folds", fallos.empty,
                   "sin fallos" if fallos.empty else fallos[["diseno", "fold", "modelo", "estado"]].to_string()))
    met["rmse_usd"] = a_usd(met["rmse"], cfg)
    met["mae_usd"] = a_usd(met["mae"], cfg)
    met = met[["diseno", "modelo", "nivel", "fold", "n_entrenamiento", "n_prueba", "rmse", "mae", "r2",
               "sesgo_medio", "rmse_usd", "mae_usd", "estado"]]
    pred.to_csv(OUT / "predicciones.csv", index=False, float_format="%.10g")
    met.to_csv(OUT / "metricas.csv", index=False, float_format="%.10g")

    log.info("4. Verificación de predicciones y métricas leídas de disco")
    pred_disco = pd.read_csv(OUT / "predicciones.csv")
    met_disco = pd.read_csv(OUT / "metricas.csv")
    checks += verificar_predicciones(pred_disco, n, DISENOS_CV)
    hold = pred_disco[pred_disco["diseno"] == "holdout_original"]
    checks.append(("holdout: los cuatro modelos evaluados en las mismas 4.128 filas de prueba",
                   len(hold) == len(p_hold[0]["test"]) and set(hold["row_id"]) == set(p_hold[0]["test"]),
                   f"filas={len(hold)}"))
    checks += recalcular_metricas(pred_disco, met_disco, cfg["tolerancia_recalculo_metricas"])
    detener_si_falla(checks, "predicciones")

    log.info("5. Reproducción del split original")
    repro = reproduccion_original(met[(met["diseno"] == "holdout_original") & (met["nivel"] == "global")], X, y, cfg)
    repro.to_csv(OUT / "reproduccion_original.csv", index=False, float_format="%.10g")
    log.info("\n" + repro.to_string(index=False))

    log.info("6. Errores por segmento, celdas y distribución")
    seg = tabla_segmentos(pred, cfg)
    seg.to_csv(OUT / "errores_por_segmento.csv", index=False, float_format="%.10g")
    cel = tabla_celdas(pred[pred["diseno"].isin(DISENOS_CV)], cfg)
    cel.to_csv(OUT / "errores_por_celda.csv", index=False, float_format="%.10g")
    dist = tabla_distribucion(pred, cfg)
    dist.to_csv(OUT / "distribucion_errores.csv", index=False, float_format="%.10g")

    log.info("7. Coeficientes e importancias (split original)")
    coef, imp = interpretacion(X, y, nombres, p_hold[0], cfg)
    coef.to_csv(OUT / "coeficientes_mco.csv", index=False, float_format="%.10g")
    imp.to_csv(OUT / "importancias.csv", index=False, float_format="%.10g")

    log.info("8. Figuras")
    figuras.generar(met, seg, cel, pred, cfg, FIG)

    log.info("9. Integridad final")
    originales_despues = verificar_originales(cfg, BASE)
    checks.append(("CSV de trabajo sin cambios (después)", sha256(BASE / cfg["datos"]["ruta"]) == cfg["datos"]["sha256"], ""))
    for o in originales_despues:
        checks.append((f"original opcional {o['archivo']} sin cambios (después)", _estado_original(o),
                       "huella coincide" if o["presente"] else "NO DISPONIBLE en esta copia; no se verifica"))

    auditoria["entorno"] = {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                            "statsmodels": sm.__version__ if hasattr(sm, "__version__") else None,
                            "plataforma": platform.platform()}
    auditoria["segundos_ejecucion"] = round(time.time() - t_inicio, 1)
    (OUT / "auditoria_datos.json").write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding="utf-8")
    detener_si_falla(checks, "final")
    conteo = pd.Series([_etiqueta(ok) for _, ok, _ in checks]).value_counts()
    log.info(f"Ejecución completa en {auditoria['segundos_ejecucion']} s: PASS={conteo.get('PASS', 0)}, "
             f"FAIL={conteo.get('FAIL', 0)}, NO DISPONIBLE (N/A)={conteo.get(NA, 0)}")


def _estado_original(o):
    """None marca NO DISPONIBLE: un original opcional ausente no se cuenta como comprobación superada."""
    return True if o["estado"] == "PASS" else None


def _etiqueta(ok):
    # "NO DISPONIBLE" (N/A) en lugar de "N/A" literal, que pandas leería como valor faltante.
    return NA if ok is None else ("PASS" if ok else "FAIL")


def detener_si_falla(checks, etapa):
    """Escribe outputs/verificaciones.csv con estados PASS/FAIL/NO DISPONIBLE y se detiene ante cualquier FAIL.

    Los fallos de integridad previos a la primera llamada (huella del CSV, originales alterados, estructura)
    detienen la ejecución antes de este punto: quedan solo en outputs/ejecucion.log y en el código de salida 1.
    """
    tabla = pd.DataFrame([(c, _etiqueta(ok), d) for c, ok, d in checks], columns=["comprobacion", "estado", "detalle"])
    tabla.to_csv(OUT / "verificaciones.csv", index=False)
    malos = tabla[tabla["estado"] == "FAIL"]
    if not malos.empty:
        log.error(f"Comprobaciones fallidas en la etapa '{etapa}':\n{malos.to_string(index=False)}")
        raise ErrorIntegridad(f"{len(malos)} comprobaciones fallidas en '{etapa}'")


def ejecutar():
    """Punto de entrada: devuelve 0 si todo se cumplió y 1 si se detuvo por una comprobación."""
    try:
        main()
    except ErrorIntegridad as e:
        log.error(f"EJECUCIÓN DETENIDA: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(ejecutar())
