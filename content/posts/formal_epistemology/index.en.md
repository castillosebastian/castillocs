---
weight: 1
title: "Formal Epistemology"
date: 2015-03-25T10:49:29-03:00
lastmod: 2015-03-06T21:29:01+08:00
draft: false
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"
- name: "featured-image-preview"
  src: "featured-image-preview.jpg"

tags: ["Knowledge"]
categories: ["Formal Epistemology"]
lightgallery: true
toc:
  auto: false
---

Formal epistemology is an interdisciplinary field that reflects on knowledge and learning using formal methods. 

<!--more-->


# Responsible AI

The pervasive application of machine learning in many sensitive environments to make important and life-changing decisions, has heightened concerns about the fairness and ethical impact of these technologies. More importantly, experiments that unveil biases and disparities inherent in these implementations have dismantled the idea of algorithmic 'neutrality', emphasizes the critical need for alignment with laws and values pertaining to human dignity.

Ethical concepts involved in this discussion:

- fairness,
- transparency,
- accauntability, and
- trust

In this setting, the concept of **responsible AI** has arisen as a vital component of every AI project. A key goal in this regard is to develop tools that can facilitate the creation of fair and ethically grounded innovations. In the subsequent sections, we will explore some of these tools, assessing their strengths and weaknesses.

The response to algorithmic bias consist in a diverse array of solutions pointing to a *responsible AI* and its many dimensions:
- explainability,
- transparency, 
- interpretability, and
- accoutability
(definiciones ver Richardson p.15)

**Fair AI: AI that is free from unintentional algorithmic bias. Fairness, as defined by Mehrabi et al. (2021), is “[T]he absence of any prejudice or favoritism toward an individual or a group based on their inherent or acquired characteristics.”

Fair AI, for the purpose of this paper, consists of solutions to combat algorithmic bias, which is often inclusive of top-tier solutions from explainability, transparency, interpretability, and accountability research (Richardson2021)

Soluctions type:
- products (i.e: software toolkit), and
- procedures.

We emphasize that prioritizing fairness in AI systems is a sociotechnical challenge. Because there are many complex sources of unfairness—some societal and some technical—it is not possible to fully “debias” a system or to guarantee fairness; the goal is to mitigate fairness-related harms as much as possible. (Microsof-Fairlear)

Challenge: 
- Conflicting Fairness Metrics (This decision is not one that should be taken lightly because the act of choosing a fairness metric is very political in that it valorizes one point of view while silencing another. Friedler et al. (2021) states that fairness experts must explicitly state the priorities of each fairness metric to ensure practitioners are making informed choices.).
- Metric Robustness: Furthermore, Friedler et al. (2018) found that many fairness metrics lack robustness. By simply modifying dataset composition and changing train-test splits, the results of their study depicted that many fairness criteria lacked stability.
- Oversimplification of Fairness A major concern in literature is the emphasis on technical solutions to algorithmic bias, which is a socio-technical problem.Madaio et al. (2020) called the sole use of technical solutions, “ethics washing,” and Selbst et al. (2019) describes the failure to account for the fact that fairness cannot be solely achieved through mathematical formulation as the “formalism trap”.
- The major challenge presented to fairness experts is translating principles and ethics codes into actionable items that practitioners and institutions can implement


- Aplicabilidad

key themes from practitioners regarding features and design considerations that would make these tools more applicable.

Applicable to a diverse range of predictive tasks, model types, and data types (Holstein et al., 2019) • Can detect & mitigate bias (Holstein et al., 2019; Olteanu et al., 2019; Mehrabi et al., 2021) • Can intervene at different stages of the ML/AI life cycle (Bellamy et al., 2018; Holstein et al., 2019; Veale & Binns, 2017) • Fairness and performance criteria agnostic (Corbett-Davies & Goel, 2018; Barocas et al., 2019; Verma & Rubin, 2018) • Diverse explanation types (Ribeiro et al., 2016; Dodge et al., 2019; Arya et al., 2019; Binns et al., 2018) • Provides recommendations for next steps (Holstein et al., 2019) • Well-supported with demos and tutorials (Holstein et al., 2019)


## Apendix: Laws

- **Europa** 

One of the most important legal documents related to AI Fairness in Europe is the **AI Act**. This act is a step closer to the first rules on Artificial Intelligence and once approved, they will be the world’s first rules on Artificial Intelligence [link](https://www.europarl.europa.eu/news/en/press-room/20230505IPR84904/ai-act-a-step-closer-to-the-first-rules-on-artificial-intelligence). The AI Act aims to ensure a human-centric and ethical development of Artificial Intelligence (AI) in Europe by endorsing new transparency and risk-management rules for AI systems [link](https://www.weforum.org/agenda/2023/03/the-european-union-s-ai-act-explained/)

The AI Act is a proposed legal framework by the European Union that aims to significantly bolster regulations on the development and use of artificial intelligence [link](https://www.caidp.org/resources/eu-ai-act/). The proposed legislation focuses primarily on strengthening rules around data quality, transparency, human oversight and accountability. It also aims to address ethical questions and implementation challenges in various sectors ranging from healthcare and education to finance and energy.

The cornerstone of the AI Act is a classification system that determines the level of risk an AI technology could pose to the health and safety or fundamental rights of a person. The framework includes four risk tiers: unacceptable, high, limited and minimal. AI systems with limited and minimal risk—like spam filters or video games—are allowed to be used with little requirements other than transparency obligations. Systems deemed to pose an unacceptable risk—like government social scoring and real-time biometric identification systems in public spaces—are prohibited with little exception.

High-risk AI systems are permitted, but developers and users must adhere to regulations that require rigorous testing, proper documentation of data quality and an accountability framework that details human oversight. AI deemed high risk include autonomous vehicles, medical devices and critical infrastructure machinery, to name a few.


- **United States** 

The United States does not have a specific AI Fairness Act equivalent to the European Union's AI Act. However, there are several initiatives and laws that aim to ensure fairness and equity in the use of AI. One such initiative is the **Blueprint for an AI Bill of Rights** by the White House Office of Science and Technology Policy. This blueprint is a guide for a society that protects all people from threats posed by AI and uses technologies in ways that reinforce our highest values [link](https://www.whitehouse.gov/ostp/ai-bill-of-rights/).

In addition, the Federal Trade Commission (FTC) has decades of experience enforcing three laws important to developers and users of AI: Section 5 of the FTC Act, the Fair Credit Reporting Act, and the Equal Credit Opportunity Act. These laws prohibit unfair or deceptive practices, including the sale or use of racially biased algorithms [link](https://www.ftc.gov/business-guidance/blog/2021/04/aiming-truth-fairness-equity-your-companys-use-ai).


{{< admonition type=info title="Formal Epistemologoy" open=false >}}
Formal epistemology reflects on knowledge and learning using formal methods. These methods not only include tools that come from logic and mathematics,[^Weinsberg] but also - and today more than ever - from computing, particularly developments in the field of artificial intelligence. This commits the reflection methodologically to certain procedures, seeking results with a level of abstraction useful for understanding complex phenomena such as knowledge and learning. Formal approach simplifies the elements and relationships under analysis, allowing epistemological problems to be productively modeled.
{{< /admonition >}}


{{< admonition type=info title="Learning" open=false >}}
Schulte notes that many results in the field of formal learning in Computer Science are linked to the notion of Valiant and Vapnik on *learning of approximately correct generalizations from a probability perspective*.[^Schulte] The *approach to correction* is closely linked to the notion of *empirical success* introduced by Gilbert Harmann, and revisited by Valiant in his reflection on the problems of induction (Valiant, 2013, Ch. 5). In any case, formal learning generally refers to a contextualized epistemological analysis where a specific empirical problem and an expected outcome in terms of learning are highlighted. This is why Schulte points out that **the majority of [formal] learning theories examine which research strategies are most reliable and efficient in generating beliefs [knowledge] about the world.** 
{{< /admonition >}}


{{< admonition type=info title="Deep Learning" open=false >}}
Deep Learning (DL) is a technique by which an agent acquires the ability to 'learn' from experience stored in the form of data. This technique is part of the field of Artificial Intelligence which, in general terms, seeks to create agents capable of performing tasks that involve complex intellectual skills, tasks such as recognizing images, processing and producing language, identifying patterns, among others. At the heart of DL is the old epistemological problem of generating 'good representations' of knowledge objects; a problem that DL solves by **representing the world as a hierarchical structure of nested concepts, where each concept is defined in relation to simpler concepts, and where the more abstract representations are computed from less abstract ones** (Goodfellow et al. 2016:8). For this reason, one of the important tasks of DL is the algorithmic transformation of concepts from simple units into complex units.[^1]
To generate representations of objects, and unlike other formal learning techniques, DL has the ability to identify defining characteristics of certain objects (*features*) and generate models (representations) from them. This ability to generate models is autonomous in a strict sense: DL does not have previous models of its objects, it constructs them using mathematical functions. To establish an analogy with humans, we might think that until not long ago, only a person could look at 10,000 photos of chairs and create a model to recognize whether a new photo (the 10,001st) is a chair or not. Now an agent that applies DL can do the same, in an amazingly fast and provenly more effective way.
The 'learning of representations' is a defining aspect of DL, and implies a simultaneous task of identifying distinctive features of objects by isolating them from particular variation factors always present in experience. For this, DL generates its complex representations (the chair model) by composing them from simple representations. The notion of 'deep learning' comes from the fact that this composition takes the form of processing at levels or layers of information.(Chollet-Allaire, Deep Learning with R, 2017.).
{{< /admonition >}}


{{< admonition type=info title="Bib" open=false >}}
- Richardson, Brianna, and Juan E. Gilbert. “A Framework for Fairness: A Systematic Review of Existing Fair AI Solutions.” arXiv, December 10, 2021. https://doi.org/10.48550/arXiv.2112.05700.
{{< /admonition >}}
