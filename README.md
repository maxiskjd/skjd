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

Comenzamos importando la base de datos y visualizándola para tener una primera idea general de cómo venía estructurada. Eso nos ayudó a entender qué tipo de información traía, cuántas variables tenía el conjunto y también a identificar cuál era la variable que queríamos predecir. En este caso, la variable objetivo es loan_status, que indica si una persona cayó en mora (1) o no (0).

Lo primero que hicimos fue revisar si había datos faltantes. Encontramos que las columnas person_emp_length y loan_int_rate tenían valores nulos: en total, 895 y 3.116 respectivamente. Como el total de datos supera las 32 mil filas, se decidió eliminar esas filas con datos nulos porque se considero que el data quedaba con una cantidad de instancias considerables para entrenar el modelo. En este tipo de casos, es posible que falte información porque algunas personas no tienen historial laboral claro o aún no se les asignó una tasa de interés.

Después analizamos gráficamente las variables usando boxplots para ver si había valores extremos que pudieran afectar negativamente al modelo. Algunas columnas, como loan_percent_income y person_emp_length, mostraban outliers bastante marcados. Para tratar estos casos, se aplicó una función que elimina los valores extremos usando el rango intercuartil, que básicamente detecta puntos que están demasiado alejados de lo que se considera el rango normal de los datos. Luego se volvieron a graficar los boxplots para verificar que estos valores extremos efectivamente ya no estuvieran.

También fue necesario transformar las variables categóricas en numéricas, ya que los modelos de machine learning necesitan trabajar solo con números. Esto se hizo usando codificación, asegurándonos de que cada categoría quedara representada de forma adecuada.

Antes de entrenar el modelo, separamos las variables predictoras (X) de la variable objetivo (y). Es decir, se eliminó loan_status del conjunto general y se trabajó con ella por separado. Luego dividimos el conjunto en dos: uno para entrenamiento y otro para testeo, dejando un 20% de los datos para el test. Además, se fijó una semilla aleatoria para que los resultados fueran reproducibles si se vuelve a correr el análisis.

Al revisar la distribución de clases en la variable objetivo, notamos un desbalance: había muchas más personas sin mora que con mora. Esto es algo esperable en este tipo de contextos, ya que normalmente la mayoría de los clientes pagan sus créditos. Por eso decidimos trabajar con Random Forest, un modelo que permite incorporar un parámetro que ajusta automáticamente los pesos de cada clase. En este caso usamos class_weight='balanced' para que el modelo no favoreciera solamente a la clase mayoritaria.

En las primeras iteraciones se probaron configuraciones más complejas, con una gran cantidad de árboles y profundidades muy altas. Al hacer estas pruebas iniciales, nos dimos cuenta de que el modelo funcionaba muy bien con los datos de entrenamiento, pero cuando lo probábamos con datos nuevos bajaba mucho el rendimiento. Eso fue una primera señal de sobreajuste. También notamos que el valor por defecto para el parámetro min_samples_leaf, que define la cantidad mínima de datos en una hoja del árbol, era muy bajo (inicialmente estaba en 1). Eso hacía que el modelo generara reglas demasiado específicas. Por eso decidimos aumentar ese valor a 80, con la idea de que cada división tuviera más respaldo y no se generaran reglas tan particulares.

Finalmente, definimos una versión más acotada del modelo: se usaron 50 árboles, con una profundidad máxima de 20. Con esa configuración, el modelo quedó mucho más balanceado y generalizó mejor.

Para verificar su comportamiento, aplicamos validación cruzada, que nos permitió evaluar el desempeño del modelo de forma más robusta. En general, los resultados fueron aceptables y se consideró que el modelo lograba un buen equilibrio: identificaba razonablemente bien los casos con mora sin perder precisión en la clase sin mora. Eso nos permitió avanzar a la siguiente etapa.

Después revisamos la curva de aprendizaje, que muestra cómo varía la precisión del modelo entre los datos de entrenamiento y validación a medida que se va entrenando. En este caso, las curvas estaban bastante cercanas, lo que es una muy buena señal. Cuando ambas curvas están alineadas o no hay una gran brecha entre ellas, significa que el modelo no está memorizando los datos, sino que está aprendiendo patrones generales que también aplica bien a nuevos casos. Por eso consideramos que no estaba sobreajustado y que estábamos en un rango óptimo para cerrar esta etapa del modelo.

Para entender mejor cómo el modelo estaba tomando decisiones, analizamos la importancia de las variables. Se generó un gráfico que muestra qué variables tenían mayor peso en la toma de decisiones del Random Forest. Por ejemplo, variables como loan_int_rate y loan_percent_income aparecían entre las más influyentes, lo cual tiene sentido considerando que se relacionan directamente con la capacidad de pago y el riesgo.

También se visualizó uno de los árboles de decisión individuales que componen el Random Forest. Esto se hizo para ver de forma más clara cómo se estaban tomando las decisiones a nivel de árbol, entendiendo mejor los umbrales y las divisiones que se hacían para llegar a una predicción. Esta parte ayuda a interpretar un poco más cómo piensa el modelo.

Por último, se generó la matriz de confusión, que permite ver en detalle cuántos casos fueron correctamente clasificados y cuántos no. Gracias a esta matriz pudimos revisar los errores más importantes, como los falsos positivos y los falsos negativos, que son especialmente relevantes cuando se trata de predecir morosidad. Estos errores tienen un impacto importante, ya que podrían llevar a otorgar un crédito a alguien con alto riesgo, o lo contrario: rechazarlo injustamente. A pesar de estos errores, la matriz mostró que el modelo tenía un rendimiento aceptable y se comportaba de forma coherente.

# Resultados

Uno de los primeros resultados relevantes fue la distribución de clases dentro del conjunto de datos. Como se mencionó anteriormente, había un fuerte desbalance: más de 17.500 casos correspondían a personas sin mora (clase 0), mientras que casi 5.000 correspondían a personas con mora (clase 1). Este desbalance nos obligó a tomar decisiones específicas al momento de entrenar el modelo, como usar el parámetro class_weight='balanced' en el Random Forest. Esta estrategia permitió que el modelo le diera más peso a los casos minoritarios, ayudando a mejorar su capacidad para identificar correctamente las personas que podrían tener dificultades de pago.

Luego de entrenar el modelo con esta estrategia, se evaluaron los resultados obtenidos usando el classification_report, que nos entrega métricas como precisión, recall y f1-score para cada clase. En términos generales, se logró una precisión alta para la clase 0 (sin mora), lo cual es esperable considerando su alta representación. En cambio, para la clase 1 (con mora), los resultados fueron más bajos, aunque dentro de rangos aceptables. Esto indica que el modelo, si bien logra identificar muchos casos de mora, también comete errores, especialmente al momento de confundir personas que sí van a caer en mora con personas que no.

En relación con iteraciones anteriores, el gráfico de la curva de aprendizaje mostró una mejora importante. En versiones previas del modelo, se observaba que la precisión en los datos de entrenamiento era casi perfecta (cerca de 1.00), mientras que en validación era bastante más baja (cercana al 0.80), lo que evidenciaba un claro sobreajuste. En cambio, en la versión actual, la precisión en entrenamiento sube gradualmente desde poco más de 0.83 hasta cerca de 0.86, y en validación desde 0.82 hasta casi 0.83. La diferencia entre ambas curvas es apenas de un 0.3%, lo que se interpreta como una señal de que el modelo no está sobreajustando y que ha aprendido de forma más natural y generalizable. Esta cercanía entre curvas refleja un mejor equilibrio y nos permite confiar más en el rendimiento del modelo con datos nuevos.

En cuanto a la importancia de las variables, se identificaron varias que jugaron un rol clave en la clasificación. Por ejemplo, loan_percent_income fue una de las más influyentes, lo cual tiene sentido porque indica qué parte del ingreso de una persona se destina al préstamo, y eso da señales claras sobre su capacidad real de pago. También apareció como importante la variable loan_int_rate, que representa la tasa de interés asignada: tasas más altas suelen asociarse a clientes considerados de mayor riesgo. Otra variable con peso fue loan_grade, que también está relacionada con el perfil crediticio asignado por la institución financiera. Finalmente, person_income en sí fue otra variable muy relevante, ya que refleja directamente la capacidad económica del solicitante. Todas estas variables tienen una lógica bastante coherente con lo que se busca predecir.

Por otro lado, también se generó la matriz de confusión, la cual muestra el rendimiento del modelo en términos de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos. Los resultados obtenidos fueron los siguientes:

- Predijo correctamente 15.550 casos en los que la persona no cayó en mora.

- Predijo correctamente 3.405 casos en los que la persona sí cayó en mora.

- Se equivocó en 2.745 casos, clasificando como "con mora" a personas que realmente no la tenían.

- También se equivocó en 1.179 casos, clasificando como "sin mora" a personas que sí la tuvieron.

Estos errores nos permiten analizar el rendimiento del modelo desde un punto de vista práctico. Los falsos negativos son especialmente críticos porque representan a personas que el modelo consideró confiables, pero que en realidad terminaron teniendo problemas de pago. En un contexto real, esto se traduce en riesgo financiero directo para la entidad que otorga los préstamos. Por otro lado, los falsos positivos también generan consecuencias: se trata de personas que fueron clasificadas como morosas sin serlo, y que por lo tanto podrían ver rechazadas sus solicitudes de crédito injustamente. En términos financieros, esto se traduce en fuga de capital, ya que se estaría dejando de otorgar créditos a personas que sí tenían capacidad de pago, afectando potencial ingreso para la entidad.

# Conclusiones
Este trabajo permitió recorrer todo el proceso de desarrollo de un modelo de machine learning aplicado a un problema del mundo financiero real: predecir el riesgo de mora en solicitudes de crédito. A lo largo de las distintas etapas —desde la limpieza y análisis exploratorio de datos, pasando por la codificación de variables, el manejo del desbalance y la elección del modelo— fuimos ajustando parámetros e iterando hasta alcanzar un desempeño satisfactorio.

Una parte fundamental del trabajo fue entender cómo impactaban las configuraciones del modelo sobre las métricas finales del classification report. Esto nos llevó a explorar distintas estrategias, como el ajuste del min_samples_leaf para controlar el sobreajuste, y a interpretar la curva de aprendizaje como herramienta para validar que el modelo generalizaba correctamente. Ese análisis fue clave para confiar en que el modelo podía funcionar con datos nuevos y aportar valor más allá del entrenamiento.

Además, pudimos identificar variables altamente influyentes, como el porcentaje del ingreso destinado al préstamo (loan_percent_income), la tasa de interés (loan_int_rate) y el ingreso anual (person_income). Reconocer la importancia de estas variables permite no solo mejorar el modelo, sino también orientar el foco de atención hacia factores clave que deberían considerar quienes toman decisiones crediticias en la práctica.

En resumen, este proyecto no solo sirvió para entrenar un modelo preciso, sino que también fue una instancia valiosa para entender cómo una herramienta de machine learning puede integrarse en procesos reales de toma de decisiones. Utilizar modelos predictivos basados en datos mejora la eficiencia y permite a las instituciones financieras anticiparse a posibles escenarios de riesgo, optimizando así su rentabilidad y reduciendo la exposición a pérdidas.