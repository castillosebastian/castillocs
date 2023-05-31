---
weight: 1
title: "Language Models"
date: 2019-01-01T21:57:40+08:00
lastmod: 2019-01-01T16:45:40+08:00
draft: true
author: "CastilloCS"
images: []
resources:
- name: "featured-image"
  src: "featured-image.png"
tags: ["LLM", "Machine Learning"]
categories: ["AI"]

lightgallery: true
---

Introduction to Lenguage Models.

<!--more-->

## Introduction

This is a note about the domain of Natural Language Processing compiled from authors that we will cite in each passage. The objective of this material is to undertake a reconstruction of this disciplinary field from an epistemological perspective, in order to enrich the reflection that is currently being produced around the technologies that use these tools. We particularly have in mind those implementations that involve the use of strategies of *formal learning* or *artificial intelligence*.

## Language Models 

Probability is a central aspect of computational language processing[1], given the ambiguous and polysemic nature of language, in addition to the fact that its means of production always suppose the presence of “noise”.

The formula of conditional probability applied in this treatment is:

*P*(*w*|*h*)

The probability of a word given its history of preceding words.

This computation extended to the history of preceding words could be of impossible resolution since the contexts of the words can be very large. Therefore, appealing to Markov's premise that the probability of a word can be satisfactorily approximated with an observation of the nearby occurrences, previous occurrences are adopted as an appropriate estimator of the probability of a given occurrence.

How is the computation of this approximation performed? The most intuitive idea may be to calculate the *maximum likelihood estimator* or MLE, the words of a corpus are vectorized, the occurrences are counted and the values are normalized in such a way that the value associated with the occurrence of each word (or feature) falls between 0 and 1 (according to probability values). The resulting ratio is called *relative frequency*.

In this way, *n-grams* can be worked where n = 2, n = 3, n = N. Applications normally use n = 3 to n = 5 (the latter when the corpus is large enough.)

## LM Limitations

The characteristics of this strategy determine certain restrictions on its effectiveness. First of all, LM has a high dependence on the training set, so it is not very generalizable. This implies that its effectiveness is always conditioned by the similarity of genres or language domains.

Another restriction is given by the underestimation of terms whose occurrence is 0 in the training set but are frequent within the language domain in question.

Finally, it is not uncommon in many language domains the existence of open vocabularies and the occurrence, in such a case, of unknown words.

These different restrictions can be dealt with in different ways to achieve a more flexible model when assigning probability. It implies different ways of *smoothing* the probability function assigning slightly higher values to 0. For example, starting from a frequency count that is by default 1.

## LM Evaluation

The best way to evaluate the performance of an n-grams model is through concrete implementation and the resolution of a practical case. This type of evaluation is called extrinsic evaluation (end-to-end) and contrasts with intrinsic evaluation using a performance metric. This metric for language models is called *perplexity* and is calculated for each n-grams model as the inverse of the probability of the test_set normalized by the number of words (or vocabulary). As the relationship is inverse, the larger the conditional probability values of the n-grams set, the lower the perplexity. Maximizing the probability will lead to minimizing perplexity.

## Practice

In this Google Colab notebook, we perform an exercise of applying *n-grams* in Python following the *Natural Language Processing* course from the *Advanced Machine Learning Specialization* dictated through Coursera by *National Research University Higher School of Economics*, Russia:
[link](https://colab.research.google.com/drive/15mPt4LS1la1