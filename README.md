# Proyecto Predicción para solicitar Crédito
## Definición del problema
Este proyecto tiene como objetivo desarrollar un modelo de clasificación para predecir el estado futuro de los préstamos otorgados por una institución financiera. Es decir, se busca anticipar si un crédito caerá en incumplimiento o será pagado correctamente, utilizando un dataset obtenido de kaggle.

En el área financiera, el incumplimiento de pagos representa uno de los principales riesgos operacionales y financieros, afectando tanto la liquidez como la estabilidad de la institución. Por lo que aplicar técnicas de machine learning en este contexto permite identificar patrones en los datos que no son fácilmente detectables con métodos tradicionales. Esto ayuda a anticipar posibles incumplimientos antes de que ocurran, lo que da margen para actuar con tiempo al aplicar medidas concretas de mitigación, como ajustar condiciones del préstamo, exigir garantías adicionales o incluso rechazar solicitudes con alto riesgo de impago.


# Plan de acción
### 1. Dataset a utilizar
Este conjunto de datos contiene información sobre solicitantes de préstamos y las características de sus créditos, con el objetivo principal de evaluar el riesgo de incumplimiento. 

Data a utilizar:


| Nombre                    | Descripción                          |
|-----------------------------------|--------------------------------------------------|
| `person_age`                      | Edad                                             |
| `person_income`                   | Ingreso anual                                    |
| `person_home_ownership`           | Tipo de propiedad de vivienda                   |
| `person_emp_length`               | Años de experiencia laboral                      |
| `loan_intent`                     | Propósito del préstamo                           |
| `loan_grade`                      | Calificación del préstamo                        |
| `loan_amnt`                       | Monto del préstamo                               |
| `loan_int_rate`                   | Tasa de interés del préstamo                     |
| `loan_status (TARGET (y))`                     | Estado del préstamo (0: sin mora, 1: en mora)    |
| `loan_percent_income`            | Porcentaje del ingreso destinado al préstamo     |
| `cb_person_default_on_file`       | Historial de mora previa                         |
| `cb_preson_cred_hist_length`      | Antigüedad del historial crediticio             |

La variable clave es loan_status, que indica si el préstamo fue pagado (0) o entró en default (1). Estos datos permiten analizar patrones de riesgo, para que el modelo a desarrollar pueda ayudar a tomar decisiones de si dar o no un prestamo en instituciones financieras.

### 2. Preprocesamiento de datos
Los datos a utilizar en el modelo pasaran por una limpieza que incluye eliminar valores nulos o features que no aporten mucha información para así trabajar con una buena calidad de datos y no tener problemas al momento de entrenar el modelo. Dado que el dataset contiene variables categóricas relevantes (como person_home_ownership, loan_intent, loan_grade y cb_person_default_on_file), será necesario convertirlas a un formato numérico para que puedan ser trabajables.

### 3. Manejo del desbalance de clases
El dataset está desbalanceado ya que hay muchos más préstamos que fueron pagados que los que entraron en incumplimiento. Se evaluarán estrategias para compensar ese desequilibrio, considerando la opción de ajustar los pesos de las clases directamente en el modelo.
### 4. Entrenamiento y evaluación
El modelo se entrenará utilizando validación cruzada para asegurar la estabilidad de los resultados. Además de la precisión, se prestará especial atención a métricas como recall, F1-score y matriz de confusión para evaluar la cantidad de falsos positivos y falsos negativos, pues el aprobar créditos a futuros clientes morosos o rechazar créditos a potenciales pagadores, afecta directamente la rentabilidad de la institución.


## Justificación del modelo

Para este proyecto se eligió Random Forest como modelo base por varias razones. En primer lugar, se trata de un modelo robusto y versátil, capaz de manejar adecuadamente tanto variables numéricas como categóricas sin requerir transformaciones complejas.
Además, Random Forest logra un buen desempeño en tareas de clasificación, incluso en datas desbalanceados. 

Además tiene la ventaja de estimar la importancia de las variables, lo que resulta útil en áreas financieras, donde muchas veces es necesario justificar las decisiones del modelo.

También es un algoritmo que permite controlar el sobreajuste de forma efectiva mediante parámetros como la profundidad máxima de los árboles y el número mínimo de observaciones por hoja lo que ayuda a evitar el overfitting.

Aún así hay cietas limitaciones en el modelo de Random Forest. Ya que si bien permite obtener una medida de la importancia de las variables, sigue siendo en gran parte una "black box", lo cual puede dificultar explicar de forma clara por qué un préstamo en particular no fue concedido. Lo que puede ser una desventaja en el sector financiero, donde a menudo se exige justificar las decisiones ante clientes o entes reguladores.

# Medología aplicada

Comenzamos importando la base de datos y visualizándola para tener una primera idea de las características (features), sus valores y la cantidad de variables (dimensionalidad) que tiene la data. En esta etapa, identificamos la variable que queremos predecir: loan status, que representa si una persona cayó en mora (1) o no (0). 

Luego verificamos si había datos nulos. Encontramos que dos columnas, person_emp_length y loan_int_rate, tenían 895 y 3116 valores nulos, respectivamente. Es posible que en estos campos falte información porque, por ejemplo, algunas personas no tienen historial laboral claro o no se les asignó una tasa de interés al momento de consultar un préstamo. Como el conjunto de datos es bastante grande (más de 32.000 filas), decidimos eliminar esas filas sin perder representatividad.

Después hicimos un análisis gráfico con boxplots para ver si había valores extremos que pudieran afectar al modelo. Efectivamente, algunas columnas como loan_percent_income y person_emp_length mostraban outliers muy marcados. Para tratar esto, aplicamos una función que elimina estos valores extremos basándose en el rango intercuartílico.

Volvimos a graficar los boxplots para confirmar que los valores extremos hayan sido eliminados. 

A continuación, transformamos las variables categóricas a numéricas para que el modelo pueda procesarlas. También revisamos si había desbalance de clases en la variable objetivo, y confirmamos que sí había, con más de 17.500 casos sin mora y menos de 5.000 con mora. Algo que tiene sentido, ya que generalmente la mayoría de los clientes suelen pagar sus créditos. 

Debido a la categórica binaria y desbalanceada, decidimos utilizar Ramdon Forest, que cuenta con parámetros para manejar el desequilibrio ajustando los pesos de cada clase para no favorecer a la mayoritaria.

Luego preparamos los datos para el modelo, separando las variables predictoras de la clase categórica. También dividimos el conjunto de entrenamiento y testeo, con un 20% del total del dataset para el testeo, usando una semilla para garantizar resultados reproducibles.

Creamos el modelo de Random Forest, y usamos el parámetro class_weight='balanced' para que el modelo tome en cuenta el desbalance al momento de entrenar.

Una vez entrenado el modelo, evaluamos su rendimiento utilizando validación cruzada y calculamos las métricas de desempeño.

Según los resultados obtenidos,el modelo cuenta con una buena precisión general, pero notamos que el Recall (0.65)  para la clase Positiva (1)  es relativamente bajo, lo que indica que se escapan bastantes casos con mora que fueron clasificados sin mora.

Realizaremos una nueva actualización de parámetros para mejorar las métricas como el recall y el f1 score. También es de nuestro interés investigar si el modelo está preajustado, por lo que graficamos la curva de aprendizaje. El gráfico resultante muestra una precisión casi perfecta en entrenamiento pero mucho más baja en Validación, lo cual confirma que el modelo está sobreajustado a los datos de entrenamiento.Por eso exploramos estrategias para reducir este sobreajuste y mejorar el desempeño en datos no conocidos.


Para comenzar a abordar el problema de sobreajuste que detectamos en la curva de aprendizaje, decidimos analizar la importancia de las variables, para ello graficamos la importancia de cada una según el modelo Random Forest. Esto nos permite ver qué columnas están aportando más a la predicción.
Como se muestra en el gráfico, algunas variables como cb_person_default_on_file y cb_person_cred_hist_length tienen una importancia muy baja. Por otro lado, también revisamos la matriz de correlación entre variables y detectamos que loan_grade está fuertemente correlacionada con loan_int_rate, por lo que decidimos eliminarla para evitar redundancia.

Con esas columnas fuera, volvimos a entrenar el modelo, esta vez ajustamos los hiperparametros: usamos solo 30 árboles y una profundidad máxima de 10 niveles.Esta decisión se tomó para limitar la complejidad del modelo y así evitar que aprenda patrones demasiado específicos del entrenamiento, lo que suele provocar sobreajuste.

Luego volvimos a evaluar las métricas del modelo, esta vez las métricas siguen siendo buenas para la clase 0, mientras que para la clase 1 vemos una mejora en el recall, lo que indica que el modelo logra identificar más casos de mora correctamente.
Para confirmar que esta versión del modelo mejora el problema de sobreajuste, volvimos a graficar la curva de aprendizaje. A diferencia de antes, donde la precisión del entrenamiento era extremadamente alta y la de validación mucho más baja, ahora ambas curvas están más cercanas entre sí. Esto muestra que el modelo ya no está memorizando los datos de entrenamiento, sino que generaliza mejor, lo cual es justamente lo que buscamos.

Finalmente, graficamos la matriz de confusión. Aunque sigue habiendo una cantidad considerable de casos con mora que son clasificados como si no la tuvieran (1598 instancias), el modelo mejora en comparación con la versión anterior. Y dado que se logró reducir el sobreajuste y mejorar la capacidad de generalización, decidimos dejar esta versión del modelo como la definitiva para este análisis.
También visualizamos uno de los árboles del bosque generado para entender mejor cómo el modelo está tomando decisiones. Esto nos ayuda a interpretar qué variables son más determinantes en cada caso y qué cortes realiza el modelo a la hora de clasificar.


