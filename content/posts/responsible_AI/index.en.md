---
weight: 1
title: "Responsible AI Tools"
date: 2015-03-25T10:49:29-03:00
lastmod: 2015-03-06T21:29:01+08:00
draft: true
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"
- name: "featured-image-preview"
  src: "featured-image-preview.jpg"

tags: ["fairness", "bias"]
categories: ["Responsible AI"]
lightgallery: true
toc:
  auto: false
---

Responsible AI has emerged as an essential part of every Machine Learning project, emphasizing the pursuit of tools that foster the development of fair and ethically sound innovations.

<!--more-->


# Unfearness and Responsible AI

The pervasive application of machine learning in many sensitive environments to make important and life-changing decisions, has heightened concerns about the fairness and ethical impact of these technologies. More importantly, experiments that unveil biases and disparities inherent in these implementations have dismantled the idea of algorithmic 'neutrality', emphasizes the critical need for alignment with laws and values pertaining to human dignity.

In this setting, the concept of **responsible AI** has arisen as a vital component of every AI project. A key goal in this regard is to develop tools that can facilitate the creation of fair and ethically grounded tools. In the subsequent sections, we will explore some of these tools, assessing their strengths and weaknesses.


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


{{< admonition type=info title="X" open=false >}}

{{< /admonition >}}


{{< admonition type=info title="Y" open=false >}}

{{< /admonition >}}


{{< admonition type=info title="Z" open=false >}}

{{< /admonition >}}


{{< admonition type=info title="Bib" open=false >}}
- Mehrabi, Ninareh, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. “A Survey on Bias and Fairness in Machine Learning.” arXiv, January 25, 2022. https://doi.org/10.48550/arXiv.1908.09635.
- Richardson, Brianna, and Juan E. Gilbert. “A Framework for Fairness: A Systematic Review of Existing Fair AI Solutions.” arXiv, December 10, 2021. https://doi.org/10.48550/arXiv.2112.05700.
{{< /admonition >}}
