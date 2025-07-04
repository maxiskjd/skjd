# Proyecto Predicción para solicitar Crédito
## Definición del problema
El presente proyecto tiene como objetivo desarrollar un modelo de clasificación para predecir el estado futuro de los préstamos otorgados por una institución financiera. En concreto, se busca anticipar si un crédito caerá en incumplimiento o será pagado correctamente, utilizando variables disponibles en el momento de su evaluación, como antecedentes del solicitante, características del préstamo y situación financiera declarada.

En el sector financiero, el incumplimiento de pagos representa uno de los principales riesgos operacionales y financieros, afectando tanto la liquidez como la estabilidad de la institución. Por esta razón, la implementación de herramientas analíticas basadas en modelos predictivos permite fortalecer los procesos de gestión del riesgo crediticio y/o optimizar la asignación de recursos.

Aplicar técnicas de machine learning en este contexto permite identificar patrones en los datos históricos que no son fácilmente detectables con métodos tradicionales. Esto ayuda a anticipar posibles incumplimientos antes de que ocurran, lo que da margen para actuar con tiempo. Con predicciones más precisas, la institución puede aplicar medidas concretas de mitigación, como ajustar condiciones del préstamo, exigir garantías adicionales o incluso rechazar solicitudes con alto riesgo de impago.

# Plan de acción
### 1. Dataset a utilizar
Este conjunto de datos contiene información sobre solicitantes de préstamos y las características de sus créditos, con el objetivo principal de evaluar el riesgo de incumplimiento. Incluye variables como la edad, el ingreso anual, la situación de vivienda y el historial laboral de los solicitantes, así como detalles del préstamo, como el monto, la tasa de interés, el propósito y la calificación crediticia. También registra si el solicitante ha tenido incumplimientos previos y la longitud de su historial crediticio.

La variable clave es loan_status, que indica si el préstamo fue pagado (0) o entró en default (1). Estos datos son útiles para analizar patrones de riesgo, desarrollar modelos predictivos y optimizar decisiones de aprostamo en instituciones financieras.

### 2. Preprocesamiento de datos
En esta etapa, se prepararán los datos para el modelado mediante técnicas de transformación. Dado que el dataset contiene variables categóricas relevantes (como person_home_ownership, loan_intent, loan_grade y cb_person_default_on_file), será necesario convertirlas a un formato numérico para que puedan ser procesadas por los algoritmos de machine learning.

Además, se aplicará normalización o escalamiento a las variables numéricas cuando sea necesario, y se manejarán los valores nulos de manera adecuada para garantizar la calidad de los datos y evitar errores durante el entrenamiento del modelo.


