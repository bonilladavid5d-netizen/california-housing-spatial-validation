# Nota metodológica

Complementa a `PROTOCOLO.md`, que fija las decisiones antes de ejecutar. Aquí se documenta cómo se implementaron y cómo leer los resultados. Todas las cifras provienen de `outputs/`.

## 1. Definiciones

| Variable | Definición | Agregación | Unidad |
|---|---|---|---|
| `MedHouseVal` (objetivo) | Valor de la vivienda en el *block group* | **mediana** | cientos de miles de USD de 1990 |
| `MedInc` | Ingreso de los hogares | **mediana** | decenas de miles de USD de 1990 |
| `HouseAge` | Edad de las viviendas | **mediana** | años |
| `AveRooms` | Habitaciones por hogar | promedio | habitaciones |
| `AveBedrms` | Dormitorios por hogar | promedio | dormitorios |
| `AveOccup` | Personas por hogar | promedio | personas |
| `Population` | Población del área | total | personas |
| `Latitude`, `Longitude` | Localización del área | coordenada | grados |

Que el objetivo sea una mediana importa: describe la vivienda típica del área, no un promedio de precios ni una transacción. Los promedios por hogar (`AveRooms`, `AveOccup`) toman valores extremos en áreas con pocos hogares o con población institucional: 69 áreas tienen más de 20 habitaciones por hogar y 37 más de 10 personas por hogar. No se recortan; los árboles son poco sensibles a ellos y MCO los incorpora tal cual, como en el trabajo original.

**Escala en USD.** Error en USD = error en escala original × 100.000. Son dólares nominales de 1990; no se ajustan por inflación.

**Techos.** El objetivo está limitado en 5,00001 (965 filas). Además hay 27 filas con valor exactamente 5,0, que no están en el techo. `HouseAge` está limitada en 52 (1.273 filas) y `MedInc` en 15,0001 (49 filas). Se usan tal como vienen.

**Coordenadas repetidas.** Hay 12.590 pares de coordenadas únicos. 12.403 filas comparten coordenadas con al menos otra fila (máximo 15 por punto). Las filas con la misma coordenada caen siempre en la misma celda y, por tanto, en el mismo fold espacial.

## 2. Particiones

**Split original.** `train_test_split(test_size=0.2, random_state=123)` sobre los índices 0…20.639. Las filas de entrenamiento se pasan a los modelos en el orden que devuelve la función, como en el notebook: el bootstrap de Random Forest depende del orden de las filas, y ordenarlas cambia el resultado en el cuarto decimal.

*Observación:* el conjunto de prueba del split original coincide exactamente con el fold 1 del diseño aleatorio, porque ambas funciones usan la misma permutación con semilla 123. Por eso las métricas de MCO, Gradient Boosting y la media coinciden en ambos. Random Forest difiere ligeramente (0,4964 frente a 0,4970) solo por el orden de las filas de entrenamiento.

**A. Aleatorio.** `KFold(5, shuffle=True, random_state=123)`. Cada fold tiene 4.128 filas de prueba y 16.512 de entrenamiento.

**B. Espacial.**

1. Celda = `(floor(lat/0,5), floor(lon/0,5))`. Resultan 176 celdas.
2. `GroupKFold(5)` con la celda como grupo. Cada fold de prueba tiene entre 33 y 37 celdas y 4.128 filas.
3. **Margen.** Para cada fold se calcula la distancia haversine (radio 6.371,0088 km) de cada fila de entrenamiento a la fila de prueba más cercana, con un `BallTree`. Se excluyen las que están a menos de 10 km.

| Fold espacial | Entrenamiento antes | Excluidas por margen | Entrenamiento final | % excluido | Distancia mínima resultante |
|---|---|---|---|---|---|
| 1 | 16.512 | 3.216 | 13.296 | 19,5 % | 10,006 km |
| 2 | 16.512 | 3.368 | 13.144 | 20,4 % | 10,007 km |
| 3 | 16.512 | 944 | 15.568 | 5,7 % | 10,008 km |
| 4 | 16.512 | 2.294 | 14.218 | 13,9 % | 10,008 km |
| 5 | 16.512 | 2.002 | 14.510 | 12,1 % | 10,002 km |

*Fuente: `outputs/tamanos_entrenamiento.csv`.*

`GroupKFold` de scikit-learn 1.6 reparte los grupos de forma determinista para equilibrar el tamaño de los folds; no usa semilla. Los folds mezclan celdas de distintas regiones. Con una delimitación aproximada por coordenadas (Área de la Bahía: lat 37,2–38,1, lon −122,6 a −121,7; Los Ángeles: lat 33,6–34,4, lon −118,7 a −117,6), el fold 3 retiene el 66 % de las áreas de la Bahía, incluida la celda San Francisco–Oakland (2.124 áreas), mientras que Los Ángeles queda repartido entre los folds 1, 2 y 4 (37 %, 32 % y 23 %). El fold 3 también contiene áreas de otras regiones, y la Bahía no queda completa en él: sus métricas (RMSE 0,757978 en MCO y 0,782520 en Random Forest) describen el fold completo, no una evaluación exclusiva de la Bahía. Estas delimitaciones son rectángulos aproximados, usados solo para describir los folds. Ningún fold falló al estimar un modelo.

**Archivos.** `outputs/asignacion_folds.csv` guarda, por fila: celda, rol en el split original, fold aleatorio, fold espacial y en qué folds espaciales fue excluida del entrenamiento por el margen.

## 3. Métricas

Para un conjunto de *n* predicciones con error *eᵢ = ŷᵢ − yᵢ*:

- **RMSE** = √(Σ eᵢ² / n). Penaliza más los errores grandes. **No** es el error absoluto promedio.
- **MAE** = Σ |eᵢ| / n. Error absoluto promedio.
- **R²** = 1 − Σ eᵢ² / Σ (yᵢ − ȳ)². Proporción de la varianza del objetivo que no queda en el error cuadrático, respecto a predecir la media de las filas evaluadas. **No** es un porcentaje de aciertos. Puede ser negativo: la media del entrenamiento tiene R² ≈ 0 en los diseños aleatorios y −0,020 en el espacial, porque la media del fold de prueba difiere de la del entrenamiento (en el fold espacial 5 llega a −0,27).
- **Sesgo medio** = Σ eᵢ / n. Positivo indica sobreestimación.

**Global frente a por fold.** Las métricas globales se calculan sobre las 20.640 predicciones fuera de muestra agrupadas, no promediando los folds. En este caso la diferencia entre el RMSE agrupado y el promedio de RMSE por fold llega a 0,005; es pequeña, pero el criterio se mantiene. El R² por fold usa la media del fold de prueba; el R² global, la media de todas las filas.

**Recalculabilidad.** `outputs/predicciones.csv` contiene, por fila y diseño: `row_id`, fold, coordenadas, observado y la predicción de los cuatro modelos. Todas las métricas de `outputs/metricas.csv` se recalculan desde ese archivo leído de disco; la diferencia máxima es de 5·10⁻¹⁰, debida al redondeo a 10 cifras significativas al guardar.

## 4. Análisis de errores

- **Por techo:** filas con objetivo 5,00001 frente al resto. En el techo, el error se mide contra 5,00001 y no contra el valor latente, que es desconocido y probablemente mayor. Por eso el error real en el segmento más caro no se puede conocer con estos datos, y la subestimación medida es una cota inferior en valor absoluto, siempre que el valor latente supere el techo.
- **Por banda de valor observado:** [0,1), [1,2), [2,3), [3,4), [4,5), [5; 5,00001) y techo. Al condicionar en el valor observado, la compresión hacia la media aparece mecánicamente en cualquier predictor imperfecto: sobreestimación abajo y subestimación arriba. La comparación útil es entre modelos dentro de cada banda.
- **Por localización:** celdas de 0,5°. El mapa muestra las 73 celdas con al menos 20 áreas; la tabla completa está en `outputs/errores_por_celda.csv`.
- **Distribución:** percentiles 50, 75, 90, 95 y 99 del error absoluto en USD, y la proporción de áreas con error menor que 25.000, 50.000 y 100.000 USD (`outputs/distribucion_errores.csv`).

## 5. Interpretación de coeficientes e importancias

- **MCO** (`outputs/coeficientes_mco.csv`): estimado sobre las 16.512 filas de entrenamiento del split original con errores HC3, robustos a heterocedasticidad, que el trabajo original documentó en los residuos. Un coeficiente es la diferencia media en el valor mediano asociada a una unidad más de la variable, manteniendo constantes las demás incluidas. No identifica efectos causales: hay variables omitidas, como la calidad del vecindario o la distancia a la costa, y las variables son agregados de área. **HC3 ajusta la heterocedasticidad, pero no la dependencia espacial entre observaciones**: áreas cercanas comparten factores no observados, así que los errores estándar, intervalos y p-valores de `coeficientes_mco.csv` no constituyen inferencia ajustada espacialmente y probablemente subestiman la incertidumbre. Se presentan como descripción; no se estimó una alternativa.
- **Importancia por impureza:** se reporta por continuidad con el notebook, pero tiende a favorecer variables continuas con muchos valores.
- **Importancia por permutación** sobre el 20 % de prueba (10 repeticiones, semilla 123): aumento del RMSE al desordenar una variable. Es información predictiva condicionada al modelo; con predictores correlacionados (latitud y longitud; habitaciones y dormitorios) se reparte de forma inestable.

## 6. Limitaciones

1. **Temporal y geográfica:** California, 1990. No hay evidencia sobre otros estados ni otras épocas.
2. **Agregación:** las conclusiones son sobre áreas, no sobre viviendas; trasladarlas a viviendas individuales sería una falacia ecológica.
3. **Techo del objetivo:** sesga a la baja las predicciones altas e impide medir el error real en el 4,7 % superior.
4. **Comparación A–B descriptiva:** B cambia a la vez la cercanía y el tamaño del entrenamiento (de 16.512 a entre 13.144 y 15.568 filas). B cambia también la composición del entrenamiento. La caída de desempeño no puede atribuirse solo a la dependencia espacial. Un control con submuestreo aleatorio del mismo tamaño queda como extensión opcional.
5. **Sensibilidad al diseño espacial:** los resultados dependen del tamaño de celda (0,5°), del margen (10 km) y de la asignación de `GroupKFold`. No se exploraron alternativas para no ajustar el diseño a los resultados.
6. **Hiperparámetros fijos:** no se calibraron. El orden entre Random Forest y Gradient Boosting podría cambiar con otra configuración.
7. **Reproducción de Random Forest:** diferencia de 3·10⁻⁵ en el RMSE frente a la referencia, no completamente explicada (sección 8).
8. **Sin inferencia sobre diferencias entre modelos:** las comparaciones de RMSE son puntuales; no se estimaron intervalos ni pruebas para las diferencias entre modelos o diseños.

## 7. Hallazgos observados e hipótesis

**Observado** (`outputs/metricas.csv`, RMSE sobre predicciones agrupadas):

| Reducción del RMSE frente a MCO | A. Aleatorio | B. Espacial |
|---|---|---|
| Random Forest | 31,0 % | 11,3 % |
| Gradient Boosting | 29,1 % | 15,2 % |

- El modelo con menor RMSE observado es Random Forest en A (0,502007) y Gradient Boosting en B (0,627286; R² 0,7044889062, es decir, 0,704 a tres decimales). Es un resultado para estas especificaciones y diseños, no una superioridad general.
- Entre A y B, el RMSE sube un 30,8 % en Random Forest, un 21,6 % en Gradient Boosting y un 1,6 % en MCO.
- En B, Gradient Boosting tiene RMSE 62.728,61 USD, MAE 45.239,03 USD y mediana del error absoluto 33.529,33 USD. Son tres métricas distintas.

**Hipótesis compatible, no identificada por el diseño.** Las coordenadas y el ingreso concentran la mayor información predictiva en los árboles (`outputs/importancias.csv`). Una explicación compatible es que parte de la ventaja de los árboles en A proviene de información local, de áreas cercanas presentes en el entrenamiento, que B retira. El diseño no permite atribuirle la diferencia, porque entre A y B cambian a la vez la cercanía geográfica, la composición regional y el tamaño del entrenamiento. Tampoco se midió la cercanía a nivel de cuadra o vivienda: las coordenadas identifican áreas censales, no viviendas.

## 8. Reproducción del trabajo original

| Evaluación (split 80/20, semilla 123) | RMSE | R² | Notebook original | Referencia independiente |
|---|---|---|---|---|
| Media de entrenamiento | 1,153189 | −0,000024 | — | 1,153189 |
| MCO en prueba | 0,719738 | 0,610455 | 0,724100 / 0,606233 *(dentro de muestra)* | 0,719738 / 0,610455 |
| Random Forest | 0,496386 | 0,814712 | 0,496417 / 0,814688 | 0,496417 / 0,814688 |
| Gradient Boosting | 0,504268 | 0,808781 | 0,504268 / 0,808781 | 0,504268 / 0,808781 |

*Fuente: `outputs/reproduccion_original.csv`.*

- **Corrección.** El notebook estimó MCO con las 20.640 filas y lo evaluó sobre esas mismas filas, mientras que los árboles se evaluaron en el 20 % de prueba. La cifra original (0,724100) se reproduce exactamente al repetir la evaluación dentro de muestra. Que el RMSE en prueba sea algo menor no implica que MCO generalice mejor: esas 4.128 filas son algo más fáciles de predecir; en A, MCO da 0,7277.
- **Diferencia residual en Random Forest (3·10⁻⁵ en RMSE).** Media, MCO y Gradient Boosting coinciden con la referencia a seis decimales; Random Forest no. Se comprobó que no la causan los datos (la copia de scikit-learn y el CSV son idénticos en `float32` y dan el mismo resultado) ni la paralelización (`n_jobs=1` y `-1` coinciden). Una hipótesis es una versión distinta de scikit-learn en el entorno original, que el notebook no registra; no se verificó. La discrepancia queda documentada como no completamente explicada y no altera ninguna conclusión.
- **Orden de las filas.** Durante el desarrollo, ordenar los índices de entrenamiento del split original cambió el RMSE de Random Forest a 0,4970, porque el bootstrap depende del orden. Se conserva el orden de `train_test_split`.

## 9. Comprobaciones y estados

`python3 run.py` registra cada comprobación en `outputs/verificaciones.csv` con uno de tres estados:

- **PASS:** comprobada y superada.
- **FAIL:** comprobada y no superada. Detiene la ejecución con código 1.
- **NO DISPONIBLE (N/A):** no se pudo comprobar porque falta un archivo opcional. No cuenta como superada ni bloquea la ejecución. Se escribe como texto `NO DISPONIBLE` porque pandas leería `N/A` como valor faltante.

Los originales del trabajo de curso (`../california_housing.csv`, `../Entrega_final_final.ipynb` y, en una copia privada, `evidencia_original/Entrega_final_final.ipynb`) son opcionales: si faltan quedan NO DISPONIBLE, y si están presentes con otra huella la ejecución se detiene. La huella del CSV de trabajo (`data/california_housing.csv`) es siempre obligatoria.

**Fallos tempranos.** Si la huella del CSV, la estructura de los datos o la huella de un original presente fallan, la ejecución se detiene antes de construir las particiones y no escribe `verificaciones.csv`. Como cada ejecución borra antes las salidas previas, la evidencia queda en `outputs/ejecucion.log` y en el código de salida 1.

**Ejecución de referencia** (carpeta de trabajo, con los dos originales de la carpeta superior presentes y sin la copia privada): 55 PASS, 0 FAIL y 2 NO DISPONIBLE. Los conteos dependen de qué archivos opcionales están presentes. Incluyen: ausencia de solapamiento entre entrenamiento y prueba; cada celda en un solo fold; distancia mínima entrenamiento–prueba ≥ 10 km, recalculada por fuerza bruta (mínimos de 10,002 a 10,008 km); una predicción por fila, modelo y diseño; mismas filas de evaluación para todos los modelos; y métricas recalculadas desde `predicciones.csv` leído de disco (diferencia máxima 5,04·10⁻¹⁰, por redondeo al guardar).
