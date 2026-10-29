"""Pruebas de componentes. Ejecutar con: python3 -m unittest discover -s tests -v"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import pipeline
from src.datos import ErrorIntegridad, cargar_datos, verificar_originales
from src.metricas import metricas
from src.particiones import celdas, distancia_minima_km, particion_espacial
from src.verificaciones import haversine_min_km, recalcular_metricas

BASE = Path(__file__).resolve().parent.parent
CFG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
R = CFG["disenos"]["espacial_5f"]["radio_tierra_km"]


class TestDistancias(unittest.TestCase):
    def test_un_grado_de_latitud(self):
        d = haversine_min_km(np.array([[34.0, -118.0]]), np.array([[35.0, -118.0]]), R)
        self.assertAlmostEqual(d[0], 111.195, places=2)

    def test_balltree_coincide_con_fuerza_bruta(self):
        rng = np.random.default_rng(0)
        a = np.column_stack([rng.uniform(32, 42, 500), rng.uniform(-124, -114, 500)])
        b = np.column_stack([rng.uniform(32, 42, 300), rng.uniform(-124, -114, 300)])
        np.testing.assert_allclose(distancia_minima_km(a, b, R), haversine_min_km(a, b, R), atol=1e-6)


class TestParticionEspacial(unittest.TestCase):
    def test_margen_y_celdas(self):
        rng = np.random.default_rng(1)
        lat, lon = rng.uniform(33, 38, 3000), rng.uniform(-122, -116, 3000)
        folds, grupo = particion_espacial(lat, lon, CFG)
        coords = np.column_stack([lat, lon])
        fold_de = np.empty(len(lat), int)
        for f in folds:
            fold_de[f["test"]] = f["fold"]
            self.assertEqual(len(np.intersect1d(f["train"], f["test"])), 0)
            self.assertGreaterEqual(haversine_min_km(coords[f["train"]], coords[f["test"]], R).min(), 10.0)
            if len(f["excluidas"]):
                self.assertLess(haversine_min_km(coords[f["excluidas"]], coords[f["test"]], R).max(), 10.0)
        for g in np.unique(grupo):
            self.assertEqual(len(np.unique(fold_de[grupo == g])), 1)

    def test_definicion_celda_con_coordenadas_negativas(self):
        i, j, cid = celdas([37.88, 37.49], [-122.23, -122.01], 0.5)
        self.assertEqual(list(i), [75, 74])
        self.assertEqual(list(j), [-245, -245])
        self.assertEqual(cid[0], "75_-245")


class TestMetricas(unittest.TestCase):
    def test_valores_a_mano(self):
        m = metricas([1, 2, 3, 4], [1, 2, 3, 6])
        self.assertAlmostEqual(m["rmse"], 1.0)
        self.assertAlmostEqual(m["mae"], 0.5)
        self.assertAlmostEqual(m["r2"], 1 - 4 / 5)

    def test_global_no_es_promedio_de_folds(self):
        import pandas as pd
        pred = pd.DataFrame({"diseno": "x", "fold": [1, 1, 2, 2], "observado": [0, 0, 0, 0],
                             "pred_m": [1, 1, 3, 3]})
        g = metricas(pred["observado"], pred["pred_m"])
        self.assertAlmostEqual(g["rmse"], np.sqrt(5))  # el promedio de folds daría 2
        met = pd.DataFrame([{"diseno": "x", "modelo": "m", "nivel": "global", "fold": np.nan, "n_prueba": 4,
                             "rmse": 2.0, "mae": g["mae"], "r2": g["r2"]}])
        res = recalcular_metricas(pred, met, 1e-9)
        self.assertFalse(res[0][1], "una métrica global mal calculada debe detectarse")


class TestDetencion(unittest.TestCase):
    def test_huella_incorrecta_detiene_la_carga(self):
        cfg = copy.deepcopy(CFG)
        cfg["datos"]["sha256"] = "0" * 64
        with self.assertRaises(ErrorIntegridad):
            cargar_datos(cfg, BASE)

    def test_original_modificado_detiene(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "orig.csv"
            f.write_text("alterado")
            cfg = copy.deepcopy(CFG)
            cfg["originales"]["archivos"] = {str(f): "0" * 64}
            with self.assertRaises(ErrorIntegridad):
                verificar_originales(cfg, BASE)

    def test_original_opcional_ausente_es_na(self):
        cfg = copy.deepcopy(CFG)
        cfg["originales"]["archivos"] = {"no_existe/Entrega_final_final.ipynb": "0" * 64}
        filas = verificar_originales(cfg, BASE)
        self.assertEqual(filas[0]["estado"], "NO DISPONIBLE")
        self.assertIsNone(pipeline._estado_original(filas[0]))
        with tempfile.TemporaryDirectory() as tmp:
            salida_original = pipeline.OUT
            pipeline.OUT = Path(tmp)
            try:
                pipeline.detener_si_falla([("dato", True, ""), ("original ausente", None, "")], "prueba")
                tabla = __import__("pandas").read_csv(Path(tmp) / "verificaciones.csv")
            finally:
                pipeline.OUT = salida_original
        self.assertEqual(list(tabla["estado"]), ["PASS", "NO DISPONIBLE"])

    def test_original_presente_alterado_en_copia_detiene(self):
        with tempfile.TemporaryDirectory() as tmp:
            copia = Path(tmp) / "evidencia_original" / "Entrega_final_final.ipynb"
            copia.parent.mkdir()
            copia.write_bytes((BASE / "data" / "california_housing.csv").read_bytes()[:1000] + b"x")
            cfg = copy.deepcopy(CFG)
            cfg["originales"]["archivos"] = {"evidencia_original/Entrega_final_final.ipynb":
                                             CFG["originales"]["archivos"]["evidencia_original/Entrega_final_final.ipynb"]}
            with self.assertRaises(ErrorIntegridad):
                verificar_originales(cfg, Path(tmp))

    def test_csv_de_trabajo_alterado_en_copia_detiene(self):
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp) / CFG["datos"]["ruta"]
            destino.parent.mkdir(parents=True)
            contenido = (BASE / CFG["datos"]["ruta"]).read_text()
            destino.write_text(contenido.replace("8.3252", "8.3253", 1))
            with self.assertRaises(ErrorIntegridad):
                cargar_datos(CFG, Path(tmp))

    def test_comprobacion_fallida_detiene(self):
        with tempfile.TemporaryDirectory() as tmp:
            salida_original = pipeline.OUT
            pipeline.OUT = Path(tmp)
            try:
                with self.assertRaises(ErrorIntegridad):
                    pipeline.detener_si_falla([("ok", True, ""), ("falla", False, "")], "prueba")
            finally:
                pipeline.OUT = salida_original


if __name__ == "__main__":
    unittest.main()
