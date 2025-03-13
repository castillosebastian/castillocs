---
weight: 1
title: "Synthetic Data and Feature Selection"
date: 2025-03-13
lastmod: 2025-03-13
draft: false
author: "CastilloCS"
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"
tags: ["Synthetic Data", "Feature Selection", "Machine Learning", "Generative AI"]
categories: ["AI"]
lightgallery: true
toc:
  auto: false
---

Synthetic data is a type of data that is created by an algorithm, rather than collected from the real world. It is a powerful tool for training machine learning models, but it can also be used to create realistic data for testing and validation.

<!--more-->

## Foundation models disruption

What happened with the emergence of Large Language Models (LLMs) that caused a profound impact in the relatively niche AI community, and how did this ripple effect extend into various spheres of public life? To understand this transformation, we must explore the roots from which LLMs have sprung and the distinctive features that set them apart from their predecessors.

At their core, LLMs are not a radical departure from the well-established principles of deep neural networks (DNNs). For decades, DNNs have been instrumental in advancing a range of technologies, from image and speech recognition to natural language processing (NLP). A notable milestone in this field dates back to 2012 with the introduction of AlexNet, a type of DNN that uses convolutions, in the ImageNet Large Scale Visual Recognition Challenge. It significantly outperformed its competitors, marking the beginning of the deep learning revolution, supported by advanced infrastructure such as GPUs. However, despite their technological significance, DNNs remained largely within the domain of specialists and enthusiasts, rarely capturing the public's imagination or expectation.

The landscape began to change as researchers and engineers focused on leveraging DNNs for language-based applications. Early successes in language technologies, such as machine translation and simple chatbots, demonstrated the potential to create tools that could tackle specific, task-oriented problems with solutions that mimicked human-like understanding and responses.    

Yet, these technologies, while impressive, often fell short of replicating the depth and nuance of human cognitive abilities in language.   

A significant leap in the evolution of AI for language processing came with the development of the Transformer architecture. Just as AlexNet had marked a turning point in image recognition by harnessing the power of Convolutional Neural Networks (CNNs), the introduction of Transformers ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), revolutionized natural language processing. This was a pivotal moment in AI's evolution, leading directly to the creation of LLMs.

LLMs were not just incremental improvements; they represented a mayor step forward. By training on vastly larger datasets and integrating a more complex and nuanced understanding of language, made possible by the Transformer architecture, LLMs began to achieve — and in many cases, surpass — human-like capabilities across a wide range of linguistic domains. 

{{< admonition type=info title="The saga of OpenAI's Generative Pre-trained Transformer" open=false >}}
The history of the evolution of the Generative Pre-trained Transformer (GPT) series by OpenAI is a important journey through recent advancements in artificial intelligence and natural language processing. From its inception, the GPT series has been at the forefront of demonstrating the emergent properties and scaling laws that underlie large language models (LLMs). 

### GPT-1: The Foundation

During the 2018, OpenAI publish the paper "Improving Language Understanding by Generative Pre-Training". The first in the series, GPT-1, laid the groundwork for utilizing unsupervised learning to pre-train a language model on a large corpus of text. It demonstrated that pre-training followed by task-specific fine-tuning could achieve remarkable performance across a variety of NLP tasks, even with fewer data for supervised training. This model set the stage for the power of transformers in language understanding and generation.

### GPT-2: Scaling Up and Emergent Abilities

One year later, a second work came to ligth with the publication of "Language Models are Unsupervised Multitask Learners". In this new experiment, PT-2 significantly increased the scale, with 1.5 billion parameters. OpenAI highlighted the model's ability to generate coherent and surprisingly nuanced text passages, perform rudimentary reading comprehension, machine translation, and even solve some types of common sense reasoning without task-specific training. One of the seminal findings from GPT-2’s development was the demonstration of emergent abilities; as the model size increased, it began to exhibit new behaviors and capabilities that were not directly taught.

### GPT-3: The Power of Scale

By the 2020, with the publication of the paper "Language Models are Few-Shot Learners" and the presentation of GPT-3, OpenAI expanded the model to an unprecedented 175 billion parameters. This leap forward demonstrated the "scaling laws" in AI, where increasing the model size, data, and compute led to predictable improvements in performance. GPT-3 showcased remarkable few-shot learning abilities, where the model could understand and perform tasks with just a few examples provided in the prompt. This model emphasized the potential of LLMs to generalize across tasks without task-specific fine-tuning, a significant breakthrough in AI research.
{{< /admonition >}}

As a result, LLMs have evolved beyond the realm of task-specific tools to become generalized platforms capable of learning and adapting to an unprecedented breadth of applications. From writing and summarizing texts to generating creative content and engaging in sophisticated dialogues, these platforms have demonstrated flexibility and depth previously unimaginable. They have transitioned from being a subject of academic interest to a cornerstone of new products and services that touch many aspects of our daily lives.  Yet, despite these advancements, LLMs face critical challenges that could limit their effectiveness, particularly when deployed in domains where precision and current knowledge are essential.

## Foudational models and the need for truthfulness 

One of the most significant issues confronting Large Language Models is their tendency to "hallucinate" or generate factually incorrect or misleading information. This limitation stems primarily from the models' design: they are trained on vast datasets of text from the internet, books, and other sources without any intrinsic mechanism to verify the truthfulness of the content they generate. Consequently, these models are susceptible to producing responses that might seem plausible but are factually incorrect, especially in expert domains such as law, finance, and healthcare where accuracy is crucial. This challenge highlights a critical gap in their ability to serve as reliable tools in experts and professional settings.

The problem of hallucination in LLMs is closely tied to another critical challenge: access to factual information. LLMs are typically static models once trained, meaning they are not updated with new information unless retrained with new data. This static nature implies that they can quickly become outdated, further compounding the issue of generating incorrect or irrelevant information based on older data sets.

The consequence of these issues is significant. In domains where expert knowledge and up-to-date information are paramount, reliance on LLMs without addressing these limitations can lead to decisions based on outdated or incorrect data, which can be detrimental in fields like healthcare, where patient outcomes can depend on the latest medical research, or in law, where legal precedents may shift over time.

Addressing the problem of hallucination and the need for access to factual information is therefore crucial for the development of LLMs that are truly useful in professional settings. One promising approach to mitigate these problems is integrating LLMs with external databases or dynamic information sources that can provide accurate, up-to-date context for the model's responses. 

## The Role of External Databases in Improving LLM Responses

The integration of external databases into the LLM workflow offers a dual advantage: it not only provides a direct link to factual and up-to-date information but also enables the model to learn from a broader and more current dataset than what was available during its initial training. By accessing external databases, LLMs can pull in the most current data relevant to a query, whether it involves recent legal statutes, current financial market trends, or the latest medical research. This capability ensures that the responses generated are not only contextually appropriate but also reflect the most accurate information available.

The RAG architecture merges the power of pre-trained language models with the precision of information retrieval systems. This pattern typically involves two main components:

1. **Retriever:** Before generating a response, the retriever component searches an external database to find relevant documents or data snippets that might contain pertinent information for answering the user's query. This process is based on similarity measures between the query and the information stored in the database.

2. **Generator:** Once the relevant information is retrieved, the generator component of the RAG takes over. This is usually a pre-trained language model, like those used in standard LLMs, which uses both the original query and the retrieved documents to generate a coherent and informed response. The generator is capable of synthesizing information from the retriever and the knowledge it has learned during pre-training, creating a response that is both accurate and contextually enriched.

## From Naive RAG to Agents 

We can trace the progression and evolution of RAG architectures through several key stages: Naive RAG, Advanced RAG, Modular RAG, and finally Agentic RAG (see:https://arxiv.org/abs/2312.10997)

**Naive RAG**   

The earliest iterations of RAG, which we term as "Naive RAG," focus primarily on the basic integration of retrieval mechanisms with generative models. In this setup, the retrieval component is straightforward, often fetching data directly related to the input query from a fixed database or set of documents. The generator then uses this retrieved information as additional context to produce responses. Although this approach enhances the model's output by grounding it in retrieved documents, it is limited by the static nature of the retrieval process and the simplicity of its integration, which can lead to suboptimal context utilization and relevance issues in the responses.

**Advanced RAG**   

Building on the foundations of Naive RAG, the "Advanced RAG" models incorporate more sophisticated retrieval techniques, such as using machine learning to improve the relevance of retrieved documents based on the query's context and the ongoing interaction history. Advanced RAG also starts to utilize feedback mechanisms where the generator's output can influence subsequent retrieval queries, creating a more dynamic interaction between the retrieval and generation processes. This allows the system to adapt more effectively to the nuances of a conversation or a complex query sequence, thereby improving the overall coherence and relevance of the generated content.

**Modular RAG**   

"Modular RAG" architectures take the flexibility of RAG systems further by decoupling the retrieval and generation components to a greater extent, allowing for interchangeable modules that can be optimized independently. This modularity enables the integration of different retrieval or generator models depending on the specific requirements of a task or domain, such as using specialized databases for medical inquiries or financial records. Modular RAG systems can dynamically switch between different modules mid-conversation, providing tailored responses based on the most appropriate data source or generation strategy for each query.

**Agentic RAG**   

The most advanced stage, "Agentic RAG," introduces significant enhancements that imbue the system with greater agency capabilities. These RAG models are designed to not only retrieve and generate information but also to make autonomous decisions about which actions to take—such as when to retrieve more information, which sources to consult, and how to interact with other systems or databases to resolve complex queries. Agentic RAG systems are characterized by their ability to function more independently and their capacity to handle tasks that require higher levels of cognitive abilities and decision-making, similar to how a human expert might operate within specific domains.

The evolution of RAG from Naive to Agentic models illustrates a trajectory towards increasingly sophisticated, adaptable, and intelligent systems capable of performing complex tasks across various domains with high levels of accuracy and reliability. These developments highlight the potential of RAG architectures to significantly enhance the practical utility of LLMs in real-world applications, especially in knowledge-intensive fields where up-to-date and precise information is crucial.

The RAG model thus operates as a feedback loop between retrieval and generation, continually refining the information it pulls from external sources to improve the relevance and accuracy of its outputs. This mechanism makes RAG particularly useful in expert domains, where precision and up-to-date knowledge are crucial.

## Notes on building RAG-type assistants

1. Core Reasoning Engine:        

At the heart of RAG-type systems are Large Language Models (LLMs) that acts as reasoning engines. This engines are enhanced by various modules that provides the necessary tools and factual information to function effectively. This setup positions the LLMs as more than just text generators; they serve as a dynamic reasoning units capable of complex generation processes when supplemented with the right tools. In this way, RAGs can be seen as a bridge between the vast knowledge stored in external databases and the reasoning capabilities of LLMs.

2. Iterative Development Approach:     

It is critical to start with a basic design and then progressively advance to more sophisticated levels. This approach allows for full control over the application's execution. Over time, as the system's capabilities are proven and its reliability is established, you can allow the RAG to take more autonomous decisions. This gradual transition is important for ensuring the system remains robust and the project's constraints (regarding response time) are met.

3. Orchestration of Responses:   
     
Effective orchestration involves ensuring seamless integration of parametric information derived from the LLM’s inherent capabilities with the knowledge retrieved during the execution. In many scenarios, the LLM alone can handle responses effectively; however, the challenge lies in integrating this capability with dynamically retrieved external data to enhance the relevance and factual groundedness of responses.    

4. Handling API Unreliability:

APIs, which serves as bridges to external data for retrieval and completion in the generation process, do not always perform as expected (or documented). Planning for unexpected outcomes and incorporating fallback mechanisms or alternatives strategies are essential. This foresight helps maintain the system's performance and ensures consistent user experiences despite external failures.    

5. Building Effective Retrievers:     

The success of a RAG-type system heavily relies on the quality of the retrieval strategy. An effective retriever is essential to provide high-quality, relevant information to the LLM. This is akin to the adage "garbage in, garbage out" —feeding the LLM with poor quality or irrelevant data will lead to poor quality outputs. Therefore, developing sophisticated retrieval mechanisms that can accurately discern and fetch pertinent information is crucial for generating high-quality responses.

{{< admonition type=note title="Bibliography" open=false >}}
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. In Advances in Neural Information Processing Systems 25 (NIPS 2012).
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. In Advances in Neural Information Processing Systems 30 (NIPS 2017).
- Yunfan Gao et al., Retrieval-Augmented Generation for Large Language Models: A Survey, 2024.
{{< /admonition >}}

Pic by <a href="https://unsplash.com/es/@hjx518756?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">McDobbie Hu</a> en <a href="https://unsplash.com/es/fotos/fotografia-de-luces-bokeh-5RgShZblKAQ?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
  