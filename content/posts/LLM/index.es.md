---
weight: 1
title: "Modelos de Lenguaje"
date: 2019-12-01T21:57:40+08:00
lastmod: 2019-01-01T16:45:40+08:00
draft: false
author: "CastilloCS"
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"
tags: ["LLM", "Machine_Learning"]
categories: ["AI"]

lightgallery: true
---

Introducción a Modelos de Lenguaje.

<!--more-->

## Introducción

Este es un apunte sobre el dominio del Procesamiento de Lenguaje Natural
elaborado a partir de autores que citaremos en cada pasaje. El objetivo
de este manterial es emprender una reconstrucción de este campo
disciplinar desde una perspectiva epistemológica, a fin de llegar a
enriquecer la reflexión que hoy se produce en torno a las tecnologías
que emplean estas herramientas. Particularmente tenemos en mente
aquellas implementaciones que suponen el uso de estrategias de
*aprendizaje formal* o *inteligencia artificial*.

## Modelos de Lenguaje 

La probabilidad es un aspecto central del tratamiento computacional del
legunaje[1], dado el caracter ambiguo y polisémico del lenguaje, además de
que sus medios de producción siempre suponen la presencia de “ruido”.

La fórmula de probabilidad condicional aplicada en este tratamietno es:

*P*(*w*|*h*)

La probabilidad de una palabra dado su historia de palabras precedentes.

Este computo extendido a la historia de palabras precedentes podría ser
de imposible resolución dado que los contextos de las palabras pueden
ser muy grandes. Por eso, apelando a la premisa de Markow sobre que la
probabilidad de una palabra puede aproximarse satisfactoriamente con una
observación de las ocurrencias próximas, se adoptan las ocurrencias
previas como un estimador apropiado de la probabilidad de una ocurrencia
dada.

¿Cómo se efectúa el cómputo de esa aproximación? La idea más intuitiva
acaso sea calcular el *estimador de máxima verosimilitud* o MLE, se
vectoriza las palabras de un corpus, se produce contando de las
ocurrencias y se normalizan los valores de tal forma de que el valor
asociado a la ocurrencia de cada palabra (o feature) caiga entre 0 y 1
(conforme valores de probabilidad). El ratio resultante se denomina
*frecuencia relativa*.

De esta forma se pueden trabajar *n-grams* donde n = 2, n = 3, n = N.
Las aplicaciones normalmente usan n = 3 a n = 5 (esto último cuando el
corpus es suficientemente grande.)

## Restricciones del LM

Las características de esta estrategia determina ciertas restricciones a
su eficacia. En primer lugar, LM presenta gran dependencia de set de
entrenamiento por lo que resulta poco generalizable. Eso implica que su
eficacia siempre está condicionada a la similitud de generos o dominios
de lenguaje.

Otro restricción está dada por la subestimación de terminos cuya
ocurrencia es 0 en el set de entrenamiento pero resultan frecuentes
dentro del dominio de lenguaje en cuestión.

Finalmente, no es extraño en muchos dominios del lenguaje la existencia
de vocabularios abiertos y la ocurrencia, en tal caso, de palabras
desconocidas.

Estas distintas restricciones pueden tratarse de distinta forma para
lograr un modelo más flexible al momento de asignar probabilidad.
Implica distintas formas de *suavizar* la función de probabilidad
asignado valores ligeramente superiores a 0. Por ejemplo partir de un
conteo de frecuencia que por defecto sea 1.

## Evaluación de LM

La mejor forma de evaluar el desempeño de un modelo n-grams es mediante
la implementación concreta y la resolución de un caso práctico. Este
típo de evaluación se denomina evaluación extrínseca (end-to-end) y
contrasta con la evaluación intrínseca mediante una métrica de
desempeño. Esta métrica para el caso de modelos de langueja se denomina
*perplejidad* y se colcula para cada modelo n-grams como la inversa de
la probabilidad del test\_set normalizado por el número de palabras (o
vocabulario). Como la relación es inversa, cuando más grande sean los
valores de probabilidad condicional del set de n-grams menor será la
perplejidad. Maximizar la probabilidad conducirá a minimizar la
perplejidad.

## Práctica

En esta notebook en google-colab efectuamos un ejercicio de aplicación
de *n-grams* en Python siguiendo al curso *Natural Language Processing*
de la especialización *Advanced Machine Learning Specialization* dictado
a través de Coursera por *National Research University Higher School of
Economics*, Rusia:
[link](https://colab.research.google.com/drive/15mPt4LS1la1hPCPcRjnSVcjP4Xb-E6A5)

[1] Dan Jurafsky and James H. Martin, Speech and Language Processing
(3rd ed. draft), work in progress, free access URL =
<https://web.stanford.edu/~jurafsky/slp3/>.
