# Feature Selection: Intro


La selección de características es un tema importante dentro del **aprendizaje automático** (**marchine learning**) debido a la dimensionalidad que presentan los datos dentro de este dominio. Frecuentemente los individuos con los que se trabaja están representados por vectores de N dimensiones (i.e.regularmente mediciones de un proceso o fenómeno), que se necesitan procesar para resolver un problema. Estas dimensiones muchas veces superan la cantidad de individuos disponibles planteando serios problemas a los algoritmos de aprendizaje, cuya complejidad (en términos de parámetros) crece conforme crece la dimensionalidad.

Ante esta situación aparece la necesidad de buscar un subconjunto de características apropiado que permita atacar el problema de manera efectiva. Si los problemas en ML crecen exponencialmente con las dimensiones de los datos (tenemos la fórmula donde cada problema (2)N, si N = 50 el problema se vuelve intratable para búsquedas exhaustivas) se vuelven importante métodos que puedan abordar la dimensionalidad. Aparecen los métodos de optimización combinatoria. 

Podemos agrupar esos métodos siguiendo tres enfoques en el diseño de la solución:

- se determina un conjunto de dimensiones y se busca maximizar una métrica de desempeño (e.g. precisión),
- se busca el conjunto más pequeño de dimensiones que satisfaga cierto umbral de desempeño,
- se busca el compromiso óptimo entre dimensionalidad y desempeño, 

<!--more-->

## Introducción

En este comentario se presenta el campo de la **epistemología formal** como rama interdisciplinaria que reflexiona sobre el conocimiento y el aprendizaje empleando métodos formales. Estos métodos no solo incluyen herramientas que vienen de la lógica y la matemática,[^Weinsberg] sino también -y hoy más que nunca- de la computación, particularmente de los desarrollos en el campo de la **inteligencia artificial**. Con todo, para avanzar en un análisis formal del conocimiento es secundario el dominio de origen de los dispositivos de análisis, basta con que asuman características formales.[^caracteristicas_formales] Con ello la reflexión se compromete metodológicamente con ciertos procedimientos, buscando resultados con un nivel de abstracción útil para comprender fenómenos complejos como el conocimiento y el aprendizaje. Siguiendo a Weinsberg clarifiquemos esta forma de análisis con un  ejemplo. 

La tarea de confirmar hipótesis científicas puede abordarse desde una visión lógica de la siguiente manera: dada la hipótesis *h* que establece que *"todos los electrónicos tienen carga negativa"*, formalizada como $\forall$x($Ex \subset Nx$), asumiríamos ante la existencia de un individuo *a* con la propiedad de *ser un electrón* que tal individuo también tiene *carga negativa*. Contrastada la existencia del caso *a* con esas propiedades se obtendría apoyo en favor de *h*, es decir, dado que verificamos *h* en un caso particular tenemos una experiencia que abala que *h* se cumple para todos los casos. Así, siguiendo a Nicod (1930) y Weinsberg **una generalización universal es confirmada por sus instancias positivas hasta tanto no se descubra un caso que la contradiga**. Por supuesto que esta afirmación deja muchas cosas sin resolver, particularmente deja abierta la pregunta sobre cuánto peso (importancia) confirmatoria tiene una instancia particular respecto de una generalización universal. Aunque podemos evitar esta pregunta otorgando un valor absoluto a un evento confirmatorio, es inevitable pensar en el valor de una generalización en términos de los casos que la misma ha explicado satisfactoriamente. Eso dejaría a la confirmación como una magnitud.

Independientemente de la resolución de estos interrogantes, el enfoque formal simplifica los elementos y relaciones bajo análisis permitiendo modelar de manera productiva problemas epistemológicos.   


## Aprendizaje Formal 

Bajo esta idea se plantean teorías acerca de cómo y bajo qué condiciones formales se genera aprendizaje a partir de observaciones. Estas teorías pueden asumir distintas formas según los objetos y problemas abordados. Por ejemplo, Schulte señala que  muchos resultados en el campo del aprendizaje formal en Ciencia de la Computación están vinculados a la noción de Valiant y Vapnik sobre *aprendizaje de generalizaciones aproximadamente correctas desde una perspectiva de probabilidad*.[^Schulte] La *aproximación a la corrección* se vincula estrechamente a la noción de *éxito empírico* introducida por Gilbert Harmann, y retomada por Valiant en su reflexión sobre los problemas de la inducción (Valiant, 2013, Ch. 5). En cualquier caso, el aprendizaje formal remite generalmente a un análisis epistemológico contextualizado donde se recorta un problema empírico puntual y un resultado esperado en términos de aprendizaje. Por esto Schulte señala que **la mayorías de las teorías de aprendizaje [formal] examinan qué estrategias de investigación resultan más confiables y eficientes para generar creencias [conocimiento] acerca del mundo.**  

## Aprendizaje Profundo

El Aprendizaje Profundo (AP) es una técnica mediante la cual un agente adquiere la capacidad de 'aprender' de la experiencia almacenada en forma de dato. Esta técnica forma parte del campo de la Inteligencia Artificial que, en líneas generales, procura crear agentes capaces de realizar tareas que suponen habilidades intelectuales complejas, tareas como reconocer imágenes, procesar y producir lenguaje, identificar patrones, entre otras.

En el centro del AP se encuentra el viejo problema epistemológico de generar 'buenas representaciones' de objetos de conocimiento; problema que el AP resuelve a través de **representar el mundo como una estructura jerárquica de conceptos anidados, donde cada concepto se define en relación a conceptos más simples, y donde las representaciones más abstractas se computan a partir de otras menos abstractas** (Goodfellow y ot. 2016:8). Por esa razón una de las tareas importantes del AP es la transformación algorítmica de conceptos de unidades simples en unidades complejas. 

Para generar representaciones de objetos, y a diferencia de otras técnicas de aprendizaje formal, el AP posee la capacidad de identificar características definitorias de ciertos objetos (*features*) y generar modelos (representaciones) a partir de ellas. Esta capacidad de generar modelos es autónoma en un sentido estricto: el AP no posee modelos previos de sus objetos, los construye mediante funciones matemáticas. Para establecer una analogía con el ser humano podríamos pensar en que hasta no hace mucho solamente una persona podía mirar 10 mil fotos de sillas y crear un modelo para reconocer si una nueva foto (la 10.001) es una silla o no. Ahora un agente que aplica AP puede hacer lo mismo, de forma asombrosamente rápida y probadamente más efectiva. 

El 'aprendizaje de representaciones' es un aspecto definitorio del AP, e implica una tarea simultanea de identificar características distintivas de objetos aislándolas de factores de variación particular siempre presentes en la experiencia. Para esto, el AP genera sus representaciones complejas (el modelo de silla) componiéndolas a partir de representaciones simples. La noción de 'aprendizaje profundo' proviene del hecho que esta composición adquiere la forma de procesamiento en niveles o capas de información.[^Chollet]


[^Weinsberg]: Weisberg, Jonathan, "Formal Epistemology", The Stanford Encyclopedia of Philosophy (Winter 2017 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/win2017/entries/formal-epistemology/>.  
[^caracteristicas_formales]: Características que -por ahora- podemos relacionar con la idea de un lenguaje explícito (semántica y sintácticamente) con reglas definidas de producción e interpretación.    
[^Chollet]: Chollet-Allaire, Deep Learning with R, 2017.
[^Schulte]: Schulte, Oliver, "Formal Learning Theory", The Stanford Encyclopedia of Philosophy (Spring 2018 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2018/entries/learning-formal/>.
