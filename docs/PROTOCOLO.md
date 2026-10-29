# Protocolo de evaluación

Versión 1.1 · 28 de octubre de 2026. Este documento y `config.json` se escribieron **antes** de ejecutar los nuevos resultados. Las decisiones aquí fijadas no se ajustan según el desempeño observado en evaluación.

## Pregunta

¿Cuánto mejora la predicción del valor mediano de la vivienda con modelos no lineales y qué desempeño conservan cuando se evita entrenar con áreas cercanas a las evaluadas?

## Datos

- Entrada única: `data/california_housing.csv`, copia byte a byte del CSV original (SHA-256 fijado en `config.json`). Si la huella no coincide, la ejecución se detiene.
- Unidad de observación: *block group* del censo de EE. UU. de 1990. No son transacciones ni avalúos.
- Objetivo: `MedHouseVal`, valor mediano de la vivienda en unidades de 100.000 USD de 1990, con techo en 5,00001.
- Predictores: los ocho originales, sin transformaciones ni variables nuevas.

## Modelos (especificación fija)

| Modelo | Especificación |
|---|---|
| Referencia | Media del objetivo en las filas de entrenamiento de cada partición |
| MCO | Mínimos cuadrados con intercepto y los ocho predictores |
| Random Forest | 300 árboles, `random_state=123`, `max_depth=None`, `min_samples_split=2`; demás parámetros por defecto de scikit-learn |
| Gradient Boosting | 300 árboles, `learning_rate=0.05`, `max_depth=3`, `random_state=123`; demás parámetros por defecto |

No hay búsqueda de hiperparámetros. Todos los modelos de una misma partición se entrenan con las mismas filas y predicen las mismas filas.

## Diseños de evaluación

**0. Reproducción del split original.** `train_test_split(test_size=0.2, random_state=123)` sobre las 20.640 filas en el orden del CSV. Los cuatro modelos se entrenan con las 16.512 filas de entrenamiento y se evalúan sobre las mismas 4.128 filas de prueba. Esto corrige la comparación original, en la que MCO se estimó y evaluó sobre la base completa.

**A. Aleatorio, 5 folds.** `KFold(n_splits=5, shuffle=True, random_state=123)`.

**B. Espacial, 5 folds con margen.**

1. Celda de cada observación: `(floor(Latitude/0.5), floor(Longitude/0.5))`.
2. `GroupKFold(n_splits=5)` con la celda como grupo: cada celda completa queda en un solo fold.
3. Para el fold *k*, se excluyen del entrenamiento las observaciones cuya distancia haversine (radio 6.371,0088 km) a la observación de prueba más cercana sea **estrictamente menor que 10 km**.
4. Se documenta el tamaño del entrenamiento antes y después del margen.
5. Si un fold no permite estimar un modelo, se registra el fallo en las salidas y la ejecución termina con error; el diseño no se modifica.

Cada observación recibe exactamente una predicción fuera de muestra por modelo en A y en B. Se guardan índices de fila, asignación de folds y exclusiones por margen.

## Métricas

- RMSE, MAE y R² por fold y sobre las predicciones fuera de muestra agrupadas.
- Las métricas globales se calculan desde las predicciones agrupadas; **no** se promedian RMSE de folds.
- R² por fold usa la media del fold de prueba; R² global usa la media de todas las filas evaluadas.
- Errores en escala original y en USD de 1990 (×100.000).

## Análisis de errores (definido de antemano)

- Las 965 observaciones en el techo se conservan en el análisis principal.
- Métricas separadas para filas en el techo y fuera de él. En el techo, el error se mide contra 5,00001, no contra el valor latente.
- Bandas de valor observado: [0,1), [1,2), [2,3), [3,4), [4,5), [5; 5,00001) y techo (=5,00001), en unidades de 100.000 USD.
- Errores por celda de 0,5° (celdas con al menos 20 observaciones para el mapa).
- Coeficientes MCO e importancias de Random Forest se presentan como asociaciones o información predictiva, no como efectos causales.

## Interpretación

- La comparación entre A y B es descriptiva: B cambia a la vez la cercanía geográfica y el tamaño efectivo del entrenamiento.
- B evalúa áreas omitidas con separación local dentro de California en 1990; no demuestra transferencia a otro estado ni a otra época.
- Las referencias de la revisión independiente son diagnósticas; se reportan las diferencias sin forzar coincidencias.

## Comprobaciones obligatorias (la ejecución se detiene si alguna falla)

1. Huella SHA-256 del CSV de trabajo (obligatoria) y de los originales opcionales, antes y después de ejecutar (ver registro de cambios 1.2).
2. Estructura de datos: 20.640 filas, nueve columnas esperadas, sin faltantes, 965 filas en el techo.
3. Ausencia de solapamiento entre entrenamiento y prueba en cada partición.
4. Diseño espacial: cada celda en un solo fold y distancia mínima entrenamiento–prueba ≥ 10 km, recalculada con un método independiente.
5. Mismas filas de evaluación para todos los modelos.
6. Una predicción fuera de muestra por observación, modelo y diseño de validación cruzada.
7. Métricas recalculadas desde `outputs/predicciones.csv` leído de disco y comparadas con `outputs/metricas.csv`.

## Registro de cambios

- **1.1 (antes de ejecutar cualquier modelo).** Al revisar la cola superior del objetivo se encontraron 27 filas con valor exactamente 5,0, distintas de las 965 en el techo 5,00001. Con las bandas de la versión 1.0 quedaban sin segmento; se añade la banda [5; 5,00001). No cambia ningún modelo, partición ni métrica.
- **1.2 (ronda de cierre, después de los resultados; solo documentación y estados).** Los originales del trabajo de curso pasan a ser explícitamente opcionales: si faltan, la comprobación queda NO DISPONIBLE (N/A) y no cuenta como superada; si están presentes con otra huella, la ejecución se detiene. No cambia datos, modelos, hiperparámetros, semilla, particiones, predicciones ni métricas.
