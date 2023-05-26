---
weight: 1
title: "Formal Epistemology"
date: 2015-03-25T10:49:29-03:00
lastmod: 2015-03-06T21:29:01+08:00
draft: false
description: "Una mirada elemental a la epistemología"
images: 
- featured-image.jpg
resources:
- name: featured-image
  src: featured-image.jpg

tags: ["knowledge"]
categories: ["formal_epistemology"]

lightgallery: true

toc:
  auto: false
---

{{< figure src="images/featured-image.jpg" title="Lighthouse (figure)" >}}

## Introduction

In this comment, the field of formal epistemology is introduced as an interdisciplinary branch that reflects on knowledge and learning using formal methods. These methods not only include tools that come from logic and mathematics,[^Weinsberg] but also - and today more than ever - from computing, particularly developments in the field of artificial intelligence. Nevertheless, to advance in a formal analysis of knowledge, it is secondary where the analysis devices originate from, as long as they assume formal characteristics.[^formal_characteristics] This commits the reflection methodologically to certain procedures, seeking results with a level of abstraction useful for understanding complex phenomena such as knowledge and learning. Following Weinsberg, let's clarify this form of analysis with an example.

The task of confirming scientific hypotheses can be approached from a logical point of view as follows: given the hypothesis h stating that "all electrons have a negative charge", formalized as $\forall$x($Ex \subset Nx$), we would assume, in the presence of an individual a with the property of being an electron, that such an individual also has a negative charge. The existence of case a with these properties would provide support in favor of h, that is, given that we verify h in a particular case, we have an experience that supports h being fulfilled in all cases. Thus, following Nicod (1930) and Weinsberg, a universal generalization is confirmed by its positive instances until a case is found that contradicts it. Of course, this statement leaves many things unresolved, particularly it leaves the question open as to how much weight (importance) a particular instance has in confirming a universal generalization. Although we can avoid this question by giving an absolute value to a confirmatory event, it is inevitable to think about the value of a generalization in terms of the cases it has satisfactorily explained. This would leave confirmation as a magnitude.

Regardless of the resolution of these questions, the formal approach simplifies the elements and relationships under analysis, allowing epistemological problems to be productively modeled."

## Formal Learning

Under this idea, theories are proposed about how and under what formal conditions learning is generated from observations. These theories can take different forms depending on the objects and problems addressed. For example, Schulte notes that many results in the field of formal learning in Computer Science are linked to the notion of Valiant and Vapnik on *learning of approximately correct generalizations from a probability perspective*.[^Schulte] The *approach to correction* is closely linked to the notion of *empirical success* introduced by Gilbert Harmann, and revisited by Valiant in his reflection on the problems of induction (Valiant, 2013, Ch. 5). In any case, formal learning generally refers to a contextualized epistemological analysis where a specific empirical problem and an expected outcome in terms of learning are highlighted. This is why Schulte points out that **the majority of [formal] learning theories examine which research strategies are most reliable and efficient in generating beliefs [knowledge] about the world.** 

## Deep Learning

Deep Learning (DL) is a technique by which an agent acquires the ability to 'learn' from experience stored in the form of data. This technique is part of the field of Artificial Intelligence which, in general terms, seeks to create agents capable of performing tasks that involve complex intellectual skills, tasks such as recognizing images, processing and producing language, identifying patterns, among others.

At the heart of DL is the old epistemological problem of generating 'good representations' of knowledge objects; a problem that DL solves by **representing the world as a hierarchical structure of nested concepts, where each concept is defined in relation to simpler concepts, and where the more abstract representations are computed from less abstract ones** (Goodfellow et al. 2016:8). For this reason, one of the important tasks of DL is the algorithmic transformation of concepts from simple units into complex units.

To generate representations of objects, and unlike other formal learning techniques, DL has the ability to identify defining characteristics of certain objects (*features*) and generate models (representations) from them. This ability to generate models is autonomous in a strict sense: DL does not have previous models of its objects, it constructs them using mathematical functions. To establish an analogy with humans, we might think that until not long ago, only a person could look at 10,000 photos of chairs and create a model to recognize whether a new photo (the 10,001st) is a chair or not. Now an agent that applies DL can do the same, in an amazingly fast and provenly more effective way.

The 'learning of representations' is a defining aspect of DL, and implies a simultaneous task of identifying distinctive features of objects by isolating them from particular variation factors always present in experience. For this, DL generates its complex representations (the chair model) by composing them from simple representations. The notion of 'deep learning' comes from the fact that this composition takes the form of processing at levels or layers of information.[^Chollet]


[^Weinsberg]: Weisberg, Jonathan, "Formal Epistemology", The Stanford Encyclopedia of Philosophy (Winter 2017 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/win2017/entries/formal-epistemology/>.  
[^formal_characteristics]: Characteristics that -for now- we can relate to the idea of an explicit language (semantically and syntactically) with defined rules of production and interpretation.    
[^Chollet]: Chollet-Allaire, Deep Learning with R, 2017.
[^Schulte]: Schulte, Oliver, "Formal Learning Theory", The Stanford Encyclopedia of Philosophy (Spring 2018 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2018/entries/learning-formal/>.