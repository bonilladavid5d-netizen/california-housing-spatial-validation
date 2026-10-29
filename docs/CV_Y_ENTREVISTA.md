# Material para CV y entrevista

Todas las cifras provienen de `outputs/metricas.csv`. Las versiones de CV usan cifras redondeadas; conviene no inflarlas ni presentarlas como diferencias estadísticamente probadas.

## Descripción para CV

**Versión larga (sección de proyectos):**

> **Valor de la vivienda en California: modelos no lineales y validación espacial** (proyecto académico en equipo de tres, reestructurado como caso de portafolio)
> - Estimé un modelo hedónico (MCO) y lo comparé con Random Forest y Gradient Boosting sobre 20.640 áreas censales de 1990, en Python (scikit-learn, statsmodels).
> - Corregí una comparación del trabajo original en la que MCO se evaluaba dentro de muestra y los árboles fuera de muestra; reproduje las cifras originales.
> - Diseñé una validación espacial (celdas de 0,5° y margen de 10 km). Frente a MCO, la reducción del RMSE pasa del 31 % al 11 % en Random Forest y del 29 % al 15 % en Gradient Boosting, y el modelo con menor RMSE observado cambia de Random Forest a Gradient Boosting.
> - Documenté la distribución del error en USD, el sesgo por banda de valor y los límites por el techo del objetivo; el flujo es reproducible con un solo comando y comprobaciones automáticas.

**Versión corta (una línea):**

> Modelo de valor de la vivienda (MCO, Random Forest, Gradient Boosting) con validación espacial: la reducción del RMSE frente a MCO cae del 31 % al 11 % (RF) y del 29 % al 15 % (GB) al evaluar áreas separadas del entrenamiento.

**Habilidades que respalda:** econometría aplicada (precios hedónicos, errores robustos a heterocedasticidad), machine learning para regresión, diseño de validación (validación cruzada agrupada con margen espacial), comunicación de límites, Python reproducible.

## Explicación de dos minutos

> El punto de partida fue un trabajo de curso: predecir el valor mediano de la vivienda en California con datos del censo de 1990, agregados por área censal. Comparábamos un modelo hedónico lineal con Random Forest.
>
> Al revisarlo encontré un problema de comparación: el modelo lineal se había evaluado sobre los mismos datos con que se estimó, y los árboles sobre datos de prueba. Lo corregí, entrenando todos los modelos con las mismas filas y evaluándolos sobre las mismas filas, y reproduje las cifras originales.
>
> Luego cambié la pregunta: ¿cuánto depende la ventaja de los modelos no lineales de cómo se evalúan? Con validación aleatoria, Random Forest reduce el RMSE un 31 % frente al modelo lineal, de unos 72.800 a 50.200 dólares de 1990. Pero la partición aleatoria permite que haya áreas cercanas en entrenamiento y prueba. Por eso diseñé una validación espacial: celdas de medio grado asignadas completas a un fold y, además, fuera del entrenamiento todo lo que estuviera a menos de 10 km de las áreas evaluadas.
>
> Con ese diseño, la reducción del RMSE frente al modelo lineal cae al 11 % en Random Forest y al 15 % en Gradient Boosting, y el modelo con menor RMSE observado pasa a ser Gradient Boosting. Es un resultado para estas especificaciones y estos diseños, no una superioridad general ni una diferencia probada estadísticamente. Una hipótesis compatible es que parte de la ventaja de los árboles depende de información local que el diseño espacial retira; pero el diseño no lo demuestra, porque también cambia la composición y el tamaño del entrenamiento.
>
> También fui explícito con los límites. Hay un techo de 500.001 dólares, así que no se puede medir el error real en las áreas más caras; todos los modelos subestiman arriba y sobreestiman abajo. Son medianas de áreas en 1990, no viviendas ni precios actuales.
>
> Lo que me llevo es metodológico: el nivel de error reportado y el modelo que parece mejor dependen del diseño de evaluación, así que el diseño debe corresponder al uso previsto y quedar fijado antes de ver resultados.

## Preguntas probables y respuestas cortas

**¿Por qué no ajustaste hiperparámetros?**
Fijé las especificaciones del trabajo original en un protocolo escrito antes de ejecutar. El objetivo era medir cuánto cambia el desempeño según el diseño de evaluación, no maximizar una métrica; ajustar con los mismos folds habría mezclado ambas cosas.

**¿La caída en el diseño espacial es por dependencia espacial?**
No se puede afirmar. El diseño espacial cambia a la vez la cercanía geográfica, la composición del entrenamiento y su tamaño (entre 13.144 y 15.568 filas en lugar de 16.512). Lo presento como comparación descriptiva. Una extensión opcional sería un control con submuestreo aleatorio del mismo tamaño.

**¿Qué pasa en el fold donde Random Forest queda por debajo de MCO?**
En el fold espacial 3, que contiene la mayor parte de las áreas que delimité como Bahía de San Francisco junto con áreas de otras regiones, el RMSE de Random Forest es 0,782520 y el de MCO 0,757978. Es el RMSE del fold completo, no una evaluación exclusiva de la Bahía.

**¿Qué significa un R² de 0,70?**
Que el error cuadrático del modelo es aproximadamente un 30 % del que tendría prediciendo la media de las áreas evaluadas. No significa que el 70 % de las predicciones sean correctas.

**¿Por qué RMSE y MAE?**
El RMSE pesa más los errores grandes; el MAE es el error absoluto promedio. En el diseño espacial, Gradient Boosting tiene un RMSE de 62.728,61 USD y un MAE de 45.239,03 USD; la mediana del error absoluto, 33.529 USD, es otra medida distinta. La diferencia entre ellas refleja una cola de errores grandes.

**¿Se puede decir que el ingreso causa precios más altos?**
No. El coeficiente indica que 10.000 dólares más de ingreso mediano se asocian con unos 44.000 dólares más de valor mediano, condicionando en las otras variables. Hay variables omitidas, son agregados de área y los errores estándar HC3 no corrigen la dependencia espacial.

**¿Sirve hoy para valorar una casa o presupuestar un error?**
No. Son medianas de áreas en 1990, con techo en 500.001 dólares. Sirve como caso metodológico sobre cómo el diseño de evaluación cambia el error medido y el modelo con menor error observado.

**¿Cuál fue tu papel?**
En el trabajo original, en equipo con Andrés Gualdrón y Luisa Perdomo, participé en el marco teórico, la planificación, el código, el análisis y la redacción. La versión de portafolio corrige la comparación, añade los diseños de validación y el análisis de errores, y empaqueta todo de forma reproducible.
