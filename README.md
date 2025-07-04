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
