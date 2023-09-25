---
weight: 1
title: "Retrieval-augmented generation for Chatbot"
date: 2023-09-01
lastmod: 2023-09-01
draft: false
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"
- name: "featured-image-preview"
  src: "featured-image-preview.jpg"
tags: ["chatbot", "llm", "question-answering"]
categories: ["Chatbot+"]
lightgallery: true
toc:
  auto: false
---

I developed a chatbot-like agent, using retrieval-augmented generation that combines a pre-trained seq2seq model with a dense document index.

<!--more-->

## Chatbot and Data Base

In the advanced domain of Natural Language Processing (NLP), large pre-trained language models have become pivotal tools, showcasing superior performance in embedding factual knowledge and achieving state-of-the-art outcomes on various downstream tasks. However, these models exhibit constraints, particularly in their ability to accurately access and manipulate knowledge. Such limitations become evident in knowledge-intensive tasks, where their efficacy can be surpassed by specialized architectures.

In this context, I constructed a prototype to explore this technology. My goal was to integrate the capabilities of pre-trained parametric memory with the expansive nature of non-parametric one. Thus, I developed a chatbot with access to a knowledge base. Utilizing retrieval-augmented generation, this system incorporates a pre-trained seq2seq model as its parametric memory and a dense vector index of documents as its non-parametric counterpart.

## Solution's Architecture

The project is challenging as it proposes the construction of a complex system, comprised of two textual data processing modules, each with its respective language models. The final system outcome is conditioned by the performance of both modules individually and their correct integration. Several experiments were conducted, adjusting the model composition and also switching the computing architecture from CPU to GPU. Two system versions were built, with a third variation for GPU processing.

>**Question-Answering Solution Process**

| Step | Description | Details |
|------|-------------|---------|
| **1. Input** | Receive Question | The system starts by receiving a question, denoted as `'p'`. |
| **2. Vectorization** | Convert Question to Vector | The question `'p'` is transformed into its vectorized form. |
| **3. Document Search** | Match Question with Documents | The vectorized question is compared against the documents in the corpus `'C'` to find potential matches. |
| **4. Identify Relevant Documents** | Select Documents `[d1,d2]`` | Based on the similarity scores, the system identifies the most relevant documents from the corpus that might contain the answer. |
| **5. Generate Answer** | Provide Response | The system generates an answer based on the content of the most similar documents. |

The solution addresses the Question-Answering problem in two steps. It first operates on a vectorized database of our corpus 'C', where each 500-character document has its associated embedding. The search is performed based on similarity between the vectorized question 'p' and the corpus documents [d1,...,dN] using their respective indices. For similarity calculation, we use the Euclidean distance between vectors with the L2 norm (vector subtraction, where the resulting vector's L2 norm gives us the distance between them: the smaller the distance, the greater the similarity). 

{{< admonition type=info title="Embeddings" open=false >}}
### Embeddings

Embeddings are a type of word representation that captures the semantic meaning of words based on their context in a high-dimensional space. They are vectors of real numbers where words with similar meanings have similar embeddings. The idea is to represent words in a dense vector space where the position of each word is learned from its context in a large dataset. One of the most popular methods to generate embeddings is through neural networks, such as Word2Vec, GloVe, and FastText.

Mathematically, an embedding for a word $w$ can be represented as:

$e(w) = [e_1, e_2, ..., e_n]$

Where:
- $e(w)$ is the embedding of the word $w$.
- $e_1, e_2, ..., e_n$ are the components of the embedding in the $n$-dimensional space.

### Euclidean Distance for Similarity Search

Euclidean distance is a measure of the straight-line distance between two points in Euclidean space. It's commonly used to measure the similarity between vectors, such as embeddings. The idea is that the closer two vectors are in this space, the more similar they are.

Given two points $P$ and $Q$ with coordinates $(p_1, p_2, ..., p_n)$ and $(q_1, q_2, ..., q_n)$ respectively in an $n$-dimensional space, the Euclidean distance $d$ between them is given by:

$d(P, Q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + ... + (p_n - q_n)^2}$

In the context of similarity search, smaller Euclidean distances indicate higher similarity. When using embeddings for similarity search, the embeddings of two words or documents are compared using the Euclidean distance to determine how semantically similar they are.

{{< /admonition >}}

After this stage, we proceed to a second step which involves generating an answer 'r' conditioned on the retrieved information. The generation takes the question 'p' and the 2 most similar documents (smallest distance), combines them into a single 'question-context' (prompt) document, and passes it to a language generation model to formulate the answer 'r'. In this final step, we are using Llama2 with 7 billion parameters, optimized for Question-Answering (chat).


{{< admonition type=info title="Large Language Models" open=false >}}
### Large Language Models (LLMs)

Large Language Models (LLMs) are a type of deep learning model designed to understand and generate human-like text. They are trained on vast amounts of text data and have the capacity to generate coherent and contextually relevant sentences over long passages. The "large" in LLMs refers to the number of parameters these models have, often ranging from hundreds of millions to tens of billions.

Mathematically, an LLM can be represented as a function $f$ that maps an input sequence of tokens $x_1, x_2, ..., x_n$ to an output sequence of tokens $y_1, y_2, ..., y_m$:

$f: (x_1, x_2, ..., x_n) \rightarrow (y_1, y_2, ..., y_m)$

Where:
- $(x_1, x_2, ..., x_n)$ is the input sequence.
- $(y_1, y_2, ..., y_m)$ is the output sequence.
- $f$ is the function representing the LLM.

### Llama 2

Llama 2 is an advanced iteration of Large Language Models developed with a focus on chat and dialogue use cases. It consists of a collection of pretrained and fine-tuned LLMs ranging in scale from 7 billion to 70 billion parameters. The fine-tuned versions, known as Llama 2-Chat, are specifically optimized for dialogue scenarios.

Key features of Llama 2 include:
- **Scale:** Models range from 7 billion to 70 billion parameters, allowing for a wide variety of applications and use cases.
- **Performance:** Llama 2 models outperform many open-source chat models on several benchmarks.
- **Fine-tuning:** Llama 2-Chat models are fine-tuned to be optimized for dialogue use cases, ensuring coherent and contextually relevant responses in conversational scenarios.
- **Safety:** Significant efforts have been made to ensure that Llama 2-Chat models are safe and provide helpful responses. This is based on human evaluations for helpfulness and safety.
- **Open Development:** A detailed description of the approach to fine-tuning and safety improvements of Llama 2-Chat is provided, enabling the community to build upon and contribute to the responsible development of LLMs.

In essence, Llama 2 represents a significant step forward in the development of LLMs, especially for chat and dialogue applications. The emphasis on safety and the detailed documentation provided aim to ensure that the community can use and further develop these models responsibly. More details in: https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/
{{< /admonition >}}

A primary challenge in implementing the described solution on a personal-type CPU (4 processors and 12GB of RAM) is the size of the models. For this reason, in Version 1 of our solution, we worked with the following specifications: for embedding operations (on 'C' and 'p') we used the all-MiniLM-L6-v2 model that produces a dense 384-dimensional vector space. For the 'r' answer generation operation, we worked with a quantized version of Llama 2, 7b, 8-bit (an exploratory attempt was made with the 13 billion parameter model, thwarted by lack of RAM). Similarity operations are performed using the FAISS library for efficient similarity search among dense vectors.

## Evaluation

To evaluate our model, several Question-Answering experiments were conducted, taking the answers provided as a solution to the Vast-Challenge competition as the ground truth. That is, we compared the answers formulated by analysts who participated in the competition with the answers provided by our solution. For this comparison, we used cosine distance.

## Experiments: CPU

Version 1 provides unsatisfactory results not only due to the answer content but also due to system latency, caused by the prolonged generation process. For example, we have this question-answer:

| **Question** | 'What is the name of the young girl who dies and what are the causes of the death?' |
|--------------|------------------------------------------------------------------------------------|
| **Analyst's Answer** | 'The name of the girl who died is Juliana Vann. The cause of death is a lingering illness, which WFA-funded doctors claimed was caused by water contamination.’ |
| **Our Model's Answer** | 'The name of the young girl who died is Elodis. The cause of death is leukemia due to benzene poisoning.' |
| **Cosine Similarity** | 0.5957 |

We see that the model fails to identify the girl, offering an answer with a similarity score of 0.59. This difficulty is associated with the retrieval module, particularly the document vectorization model. This is because, although the documents identified as similar to the question contained relevant information, they had a marginal treatment of the question's subject. Therefore, the failure in the generator model is strongly conditioned by the context that served as input. In addition to the above, the delay in formulating the answer was 272 seconds, which is unfeasible for a solution like ours.

Given these difficulties, we implemented a Version 2 of the solution with the following transformations. We replaced the all-MiniLM-L6-v2 model with the gte-base, which scored higher on the Overall MTEB English leaderboard. With this, we not only sought to improve the representation of the documents to favor the search but also the transformation of the question for similarity analysis. Several adjustments were made to the solution architecture to improve execution time, considering that 0.62% of the delay is in the generation module. With these transformations, we obtained the following results:

| **Question** | 'What is the name of the young girl who dies and what are the causes of the death?' |
|--------------|------------------------------------------------------------------------------------|
| **Analyst's Answer** | 'The name of the girl who died is Juliana Vann. The cause of death is a lingering illness, which WFA-funded doctors claimed was caused by water contamination.’ |
| **Our Model's Answer** | 'The name of the young girl who died is Juliana Vann, and the cause of her death is a lingering illness caused by water contamination according to WFA-funded doctors.' |
| **Cosine Similarity** | 0.9747 |

The improvement in the answer content was accompanied by a reduction in system latency, which lasted 163 seconds (+- 20). A series of experiments were conducted with questions that implied extractive and generative answers, with promising results in both cases. Therefore, we took a further step, reconfiguring the entire architecture to run in a GPU environment. 

## Experiments: GPU

Version 3 of the solution was readapted to run in a higher capacity environment on **Google Colab**, which implied significant changes. Although we kept the language models selected in Version 2, all processing performed in both modules was adjusted to run on GPU. Even the similarity calculation between the question and the database runs on GPU. 

To evaluate this third version, we conducted 10 question-answer experiments. Regarding the question presented above, we obtained - as expected - the same answer obtained in Version 2 but with a latency reduction of -40 seconds. This is a positive result.

In another satisfactorily resolved question, we have these results:

| **Question** | 'Where does the core of the kidnapping activities take place?' |
|--------------|---------------------------------------------------------------|
| **Analyst's Answer** | 'The kidnapping takes place at GASTech Headquarters in the southern part of Abila, Kronos.' |
| **Our Model's Answer** | 'The core of the kidnapping activities takes place in the city of Kronos.' |
| **Cosine Similarity** | 0.7201 |

More complex questions generally yielded unsatisfactory results. For example, in the following case, we formulated a question that assumes a significant level of synthesis and abstraction of the corpus documents that the model fails to resolve.

| **Question** | 'What were the motivations behind the kidnapping carried out by the more violent wing of the POK?' |
|--------------|-------------------------------------------------------------------------------------------------|
| **Analyst's Answer** | 'The more violent wing of the POK under the leadership of Mandor Vann (uncle to Isia and Juliana Vann) were motivated to kidnap members of GASTech’s leadership to exact revenge for years of pollution that GASTech’s drilling operations have inflicted on the people of Elodis. Additional motivations include GASTech’s recent IPO which resulted in massive payouts for GASTech leadership, making them ripe for ransom. Another motivation for the kidnapping is the frustration with the corruption and lax environmental regulation of the Government of Kronos, personified by GASTech.' |
| **Our Model's Answer** | 'The motivations behind the kidnapping carried out by the more violent wing of the POK are likely rooted in their anarchist ideology and a desire to create chaos and undermine the authority of the state. They may see this act as a way to further their goals of overthrowing the government and establishing a new society free from oppressive structures.' |
| **Cosine Similarity** | 0.5404 |

The model captures a general idea of the topic but does not articulate an appropriate answer. It fails to identify motives, circumstances, and people. It suffers from a systematic phenomenon in generative models related to 'hallucination'. 

{{< admonition type=info title="LLM`s challenges" open=false >}}
### Problems with Large Language Models (LLMs)

While Large Language Models (LLMs) have shown remarkable capabilities in understanding and generating human-like text, they are not without their challenges. One of the notable problems faced by LLMs is the phenomenon known as "hallucination."

### Hallucination in LLMs

Hallucination refers to the generation of information by the model that is not present or supported by the input data. In other words, the model "makes up" details or facts that are not grounded in reality or the provided context.

#### Causes:

1. **Training Data:** LLMs are trained on vast amounts of data, and they might have encountered conflicting or incorrect information during training. This can lead them to generate outputs that are not always accurate.
2. **Model Complexity:** The sheer number of parameters in LLMs can sometimes lead to overfitting, where the model might generate outputs based on patterns it "thinks" it has learned, even if they aren't valid.
3. **Lack of Ground Truth:** Unlike supervised learning where there's a clear ground truth, LLMs often operate in settings where the "correct" output is ambiguous, leading to potential hallucinations.

#### Implications:

- **Misinformation:** Hallucination can lead to the spread of false or misleading information, especially if users trust the model's outputs without verification.
- **Reliability:** For critical applications, hallucinations can undermine the reliability of the model, making it unsuitable for tasks that require high precision.

#### Solutions:

1. **Fine-tuning:** Fine-tuning the model on a narrower, domain-specific dataset can help reduce hallucinations by making the model more specialized.
2. **Prompt Engineering:** Designing prompts carefully can guide the model to produce more accurate and less hallucinatory outputs.
3. **Model Interpretability:** Developing tools and techniques to understand why a model is producing a particular output can help in identifying and mitigating hallucinations.

In conclusion, while LLMs offer significant advancements in natural language processing, it's essential to be aware of their limitations, such as hallucination. Proper understanding, fine-tuning, and continuous monitoring are crucial to harnessing their potential responsibly.
{{< /admonition >}}

However, the reason for the failure is not attributable to the results of the modules considered individually but to the solution in general. In effect, if we analyze the elements of this third experiment, we will find that the answer has a pertinent relationship with the available information, i.e., 'question + context -> answer' are acceptable, although the final result does not relate to the truth as formulated by analysts. This is because: a) the retrieval module only takes 2 documents as relevant context information due to a model limit, leaving with partial information, b) this partial information is strongly conditioned by the bias that the information has, which translates into a text with categorical and polarizing terms, and c) the decontextualization that needs to be performed for the generation instance lacks a weighting or rebalancing strategy of the corpus's structuring semantic contents and therefore reproduces a partial and decontextualized idea of its information.

A significant number of conducted experiments show critical solution failures. They particularly point to poor performance of the retrieval module. In these cases, the model responds that it 'does not have information to answer the question ('I don't know...'). That is, the modules conflict, arriving at an unacceptable result.

| **Question** | ''On what date do the microblog collection days occur'?' |
|--------------|--------------------------------------------------------|
| **Analyst's Answer** | 'January 23, 2014. The microblog collection days that include (at least) three major events related to the scenario.' |
| **Our Model's Answer** | 'I don't know the answer to that question, I'm just an AI and do not have access to the specific information you are seeking.' |

Finally, other experiments failed because they arrived at answers that deviate from the truth. In these cases, a failure in the retrieval module can also be observed.

## Python, LLama2 and HuggingFace


``` python
import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa
import torch
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv(find_dotenv())


gpu = torch.cuda.is_available()

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

if gpu: 
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-base",
                                           model_kwargs={'device': 'cuda'})
else:
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-base",
                                       model_kwargs={'device': 'cpu'})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='What were the motivations behind the kidnapping carried out by the more violent wing of the POK?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    # Setup DBQA
    start = timeit.default_timer()
    dbqa = setup_dbqa(embeddings=embeddings)
    
    response = dbqa({'query': args.input})
    end = timeit.default_timer()

    print(f'\nAnswer: {response["result"]}')
    print('='*50)

    # Process source documents
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'\nSource Document {i+1}\n')
        print(f'Source Text: {doc.page_content}')
        print(f'Document Name: {doc.metadata["source"]}')
        #print(f'Page Number: {doc.metadata["page"]}\n')
        print('='* 60)

        #

    print(f"Time to retrieve response: {end - start}")
```

{{< admonition type=note title="Bib" open=false >}}
- Touvron, H. et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288. https://arxiv.org/abs/2307.09288. 
- Lewis, P. et al. (2021). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 https://arxiv.org/abs/2005.11401. 
{{< /admonition >}}

Pic by <a href="https://unsplash.com/es/fotos/yEQ9TOaL5FM?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink">@Luke Tanis</a>