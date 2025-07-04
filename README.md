# Proyecto Predicción para solicitar Crédito
## Definición del problema
El presente proyecto tiene como objetivo desarrollar un modelo de clasificación para predecir el estado futuro de los préstamos otorgados por una institución financiera. En concreto, se busca anticipar si un crédito caerá en incumplimiento o será pagado correctamente, utilizando variables disponibles en el momento de su evaluación, como antecedentes del solicitante, características del préstamo y situación financiera declarada.

En el sector financiero, el incumplimiento de pagos representa uno de los principales riesgos operacionales y financieros, afectando tanto la liquidez como la estabilidad de la institución. Por esta razón, la implementación de herramientas analíticas basadas en modelos predictivos permite fortalecer los procesos de gestión del riesgo crediticio y/o optimizar la asignación de recursos.

Aplicar técnicas de machine learning en este contexto permite identificar patrones en los datos históricos que no son fácilmente detectables con métodos tradicionales. Esto ayuda a anticipar posibles incumplimientos antes de que ocurran, lo que da margen para actuar con tiempo. Con predicciones más precisas, la institución puede aplicar medidas concretas de mitigación, como ajustar condiciones del préstamo, exigir garantías adicionales o incluso rechazar solicitudes con alto riesgo de impago.

## Variables del Dataset de Préstamos

| Nombre original                    | Descripción                          |
|-----------------------------------|--------------------------------------------------|
| `person_age`                      | Edad                                             |
| `person_income`                   | Ingreso anual                                    |
| `person_home_ownership`           | Tipo de propiedad de vivienda                   |
| `person_emp_length`               | Años de experiencia laboral                      |
| `loan_intent`                     | Propósito del préstamo                           |
| `loan_grade`                      | Calificación del préstamo                        |
| `loan_amnt`                       | Monto del préstamo                               |
| `loan_int_rate`                   | Tasa de interés del préstamo                     |
| `loan_status (TARGET)`                     | Estado del préstamo (0: sin mora, 1: en mora)    |
| `loan_percent_income`            | Porcentaje del ingreso destinado al préstamo     |
| `cb_person_default_on_file`       | Historial de mora previa                         |
| `cb_preson_cred_hist_length`      | Antigüedad del historial crediticio             |


# Plan de acción
### 1. Dataset a utilizar
Este conjunto de datos contiene información sobre solicitantes de préstamos y las características de sus créditos, con el objetivo principal de evaluar el riesgo de incumplimiento. Incluye variables como la edad, el ingreso anual, la situación de vivienda y el historial laboral de los solicitantes, así como detalles del préstamo, como el monto, la tasa de interés, el propósito y la calificación crediticia. También registra si el solicitante ha tenido incumplimientos previos y la longitud de su historial crediticio.

La variable clave es loan_status, que indica si el préstamo fue pagado (0) o entró en default (1). Estos datos son útiles para analizar patrones de riesgo, desarrollar modelos predictivos y optimizar decisiones de aprostamo en instituciones financieras.

### 2. Preprocesamiento de datos
En esta etapa, se prepararán los datos para el modelado mediante técnicas de transformación. Dado que el dataset contiene variables categóricas relevantes (como person_home_ownership, loan_intent, loan_grade y cb_person_default_on_file), será necesario convertirlas a un formato numérico para que puedan ser procesadas por los algoritmos de machine learning.

Además, se aplicará normalización o escalamiento a las variables numéricas cuando sea necesario, y se manejarán los valores nulos de manera adecuada para garantizar la calidad de los datos y evitar errores durante el entrenamiento del modelo.

### 3. Manejo del desbalance de clases
El dataset está desbalanceado ya que hay muchos más préstamos que fueron pagados que los que entraron en incumplimiento. Se evaluarán estrategias para compensar ese desequilibrio. Considerando la opción de ajustar los pesos de las clases directamente en el modelo.
### 4. Entrenamiento y evaluación
El modelo se entrenará utilizando validación cruzada para asegurar la estabilidad de los resultados. Además de la precisión, se prestará especial atención a métricas como el recall y el F1-score, ya que son más relevantes en este caso debido al interés particular en detectar los préstamos que podrían caer en incumplimiento.


## Justificación del modelo
Para este proyecto se eligió Random Forest como modelo base por varias razones. En primer lugar, se trata de un modelo robusto y versátil, capaz de manejar adecuadamente tanto variables numéricas como categóricas sin requerir transformaciones complejas.
Además, Random Forest suele tener un buen desempeño en tareas de clasificación, incluso en contextos de desbalance de clases, como es el caso de este dataset. 
Un aspecto adicional a favor es su capacidad para estimar la importancia de las variables, lo cual resulta especialmente útil en entornos financieros, donde muchas veces es necesario justificar las decisiones del modelo.
También es un algoritmo que permite controlar el sobreajuste de forma efectiva, mediante parámetros como la profundidad máxima de los árboles (max_depth), el número mínimo de observaciones por hoja (min_samples_leaf). Estos ajustes ayudan a evitar que el modelo se adapte demasiado a los datos de entrenamiento.

Sin embargo, el modelo Random Forest también presenta algunas limitaciones que deben considerarse en el contexto de este dataset. Si bien permite obtener una medida de la importancia de las variables, sigue siendo en gran parte una "black box", lo cual puede dificultar explicar de forma clara por qué un préstamo en particular fue clasificado como riesgoso. Esto puede ser una desventaja en el sector financiero, donde a menudo se exige justificar las decisiones ante clientes o entes reguladores.
