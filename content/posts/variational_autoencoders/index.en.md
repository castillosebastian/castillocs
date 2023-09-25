---
weight: 1
title: "Generative Models: Variational Autoencoders"
date: 2023-09-01
lastmod: 2023-09-01
draft: false
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"
- name: "featured-image-preview"
  src: "featured-image-preview.jpg"
tags: ["expectation maximization", "variational autoencoders", "deep neural networks"]
categories: ["Generative Models"]
lightgallery: true
toc:
  auto: false
---

Generative models are a class of statistical models that aim to learn the underlying data distribution from a given dataset. These models provide a way to generate new samples that are statistically similar to the training data. They have gained substantial attention in various domains, such as image generation, speech synthesis, and even drug discovery.

<!--more-->

## Generative Model

Generative models are a class of statistical models that aim to learn the underlying data distribution. Given a dataset of observed samples, one starts by selecting a distributional model parameterized by $(\theta)$. The objective is to estimate $(\theta)$ such that it aligns optimally with the observed samples.The anticipation is that it can also generalize to samples outside the training set.

The optimal distribution is hence the one that maximizes the likelihood of producing the observed data, giving lower probabilities to infrequent observations and higher probabilities to the more common ones (the principle underlying this assumption is that 'the world is a boring place' -in words of Bhiksha Raj-).

## The Challenge of Maximum Likelihood Estimates (MLE) for Unseen Observations

When training generative models, a natural objective is to optimize the model parameters such that the likelihood of the observed data under the model is maximized. This method is known as **Maximum Likelihood Estimation (MLE)**. In mathematical terms, given observed data $X$, the MLE seeks parameters $\theta$ that maximize:

> $$p_\theta(X)$$

However, for many generative models, especially those that involve latent or unobserved variables, the likelihood term involves summing or integrating over all possible configurations of these latent variables. Mathematically, this turns into:

> $$p_\theta(X) = \sum_{Z} p_\theta(X,Z)$$ $$or$$ $$p_\theta(X) = \int p_\theta(X,Z) dZ$$

Computing the log-likelihood, which is often used for numerical stability and optimization ease, leads to a log of summations (for discrete latent variables) or a log of integrals (for continuous latent variables):

> $$log p_\theta(X) = \log \sum_{Z} p_\theta(X,Z)$$ $$or\$$ $$log p_\theta(X) = \log \int p_\theta(X,Z) dZ$$

These expressions are typically intractable to optimize directly due to the presence of the log-sum or log-integral operations (see the info below).

{{< admonition type=info title="Marginalization in Joint Probability" open=false >}}
## Marginalization in the Context of Joint Probability

When discussing the computation of the joint probability for observed and missing data, the term "marginalizing" refers to summing or integrating over all possible outcomes of the missing data. This process provides a probability distribution based solely on the observed data. For example, let's assume:

-   $X$ is the observed data
-   $Z$ is the missing data
-   The joint probability for both is represented as $p(X,Z)$

If your primary interest lies in the distribution of $X$ and you wish to eliminate the dependence on $Z$, you'll need to carry out marginalization for $Z$. For discrete variables, the marginalization involves the logarithm of summation, and for continuous variables, it pertains to integration. It's essential to note that these functions that includes the log of a sum o integral defies direct optimization.
{{< /admonition >}}

Can we get an approximation to this that is more tractable (without a summation or integral within the log)?

## Overcoming the Challenge with Expectation Maximization (EM)

To address the optimization challenge in MLE with latent variables, the **Expectation Maximization (EM)** algorithm is employed. The EM algorithm offers a systematic approach to iteratively estimate both the model parameters and the latent variables.

The algorithm involves two main steps:

1.  **E-step (Expectation step)**: involves computing the expected value of the complete-data log-likelihood with respect to the posterior distribution of the latent variables given the observed data.
2.  **M-step (Maximization step)**: Update the model parameters to maximize this expected log-likelihood from the E-step.

By alternating between these two steps, EM ensures that the likelihood increases with each iteration until convergence, thus providing a practical method to fit generative models with latent variables.

For E-step the **Variational Lower Bound** is used. Commonly referred to as the Empirical Lower BOund (ELBO), is a central concept in variational inference. This method is used to approximate complex distributions (typically posterior distributions) with simpler, more tractable ones. The ELBO is an auxiliary function that provides a lower bound to the log likelihood of the observed data. By iteratively maximizing the ELBO with respect to variational parameters, we approximate the Maximum Likelihood Estimation (MLE) of the model parameters. 

Let's reconsider our aim to maximize the log-likelihood of observations $x$ in terms of $q_\phi(z|x)$.

>$$\log p_\theta(x) = \log \int z p_\theta(x,z)dz$$
>$$ = \log \int z \frac{p_\theta(x,z)q_\phi(z|x)}{q_\phi(z|x)}dz$$
>$$= \log E_{z \sim q_\phi(z|x)} \left[ \frac{p_\theta(x,z)}{q_\phi(z|x)} \right]$$
>$$\geq E_z \left[ \log \frac{p_\theta(x,z)}{q_\phi(z|x)} \right] \quad \text{(by Jensen's inequality)}$$
>$$= E_z[\log p_\theta(x,z)] + \int z q_\phi(z|x) \log \frac{1}{q_\phi(z|x)} dz$$
>$$= E_z[\log p_\theta(x,z)] + H(q_\phi(z|x))$$

In the equation above, the term $H(\cdot)$ denotes the Shannon entropy. By definition, the term "evidence" is the value of a likelihood function evaluated with fixed parameters. With the definition of:

>$$L = E_z[\log p_\theta(x,z)] + H(q_\phi(z|x)),$$

it turns out that $L$ sets a lower bound for the evidence of observations and maximizes $L$ will push up the log-likelihood of $x$. 

## Variational Autoencoders (VAEs)

Variational Autoencoders are a specific type of generative model that brings together ideas from deep learning and Bayesian inference. VAEs are especially known for their application in generating new, similar data to the input data (like images or texts) and for their ability to learn latent representations of data.

**1. Generative Models and Latent Variables**

In generative modeling, our goal is to learn a model of the probability distribution from which a dataset is drawn. The model can then be used to generate new samples. A VAE makes a specific assumption that there exist some *latent variables* (or hidden variables) that when transformed give rise to the observed data.

Let $x$ be the observed data and $z$ be the latent variables. The generative story can be seen as:

1.  Draw $z$ from a prior distribution, $p(z)$.
2.  Draw $x$ from a conditional distribution, $p(x|z)$.

**2. Problem of Direct Inference**

As discussed previously, direct inference for the posterior distribution $p(z|x)$ (i.e., the probability of the latent variables given the observed data) can be computationally challenging, especially when dealing with high-dimensional data or complex models. This is because:

> $$ p(z|x) = \frac{p(x|z) p(z)}{p(x)} $$

Here, $p(x)$ is the evidence (or marginal likelihood) which is calculated as:

> $$ p(x) = \int p(x|z) p(z) dz $$

As we saw this integral is intractable for most interesting models.

**3. Variational Inference and ELBO**

To sidestep the intractability of the posterior, VAEs employ *variational inference*. Instead of computing the posterior directly, we introduce a parametric approximate posterior distribution, $q_{\phi}(z|x)$, with its own parameters $\phi$.

The goal now shifts to making this approximation as close as possible to the true posterior. This is done by minimizing the *Kullback-Leibler divergence* between the approximate and true posterior using the ELBO function.

**4. Neural Networks and Autoencoding Structure**

In VAEs, neural networks are employed to parameterize the complex functions. Specifically:

1.  **Encoder Network**: This maps the observed data, $x$, to the parameters of the approximate posterior, $q_{\phi}(z|x)$.
2.  **Decoder Network**: Given samples of $z$ drawn from $q_{\phi}(z|x)$, this maps back to the data space, outputting parameters for the data likelihood, $p_{\theta}(x|z)$.

The "autoencoder" terminology comes from the encoder-decoder structure where the model is trained to reconstruct its input data.

**5. Training a VAE**

The training process involves:

1.  **Forward pass**: Input data is passed through the encoder to obtain parameters of $q_{\phi}(z|x)$.
2.  **Sampling**: Latent variables $z$ are sampled from $q_{\phi}(z|x)$ using the *reparameterization trick* for backpropagation.
3.  **Reconstruction**: The sampled $z$ values are passed through the decoder to obtain the data likelihood parameters, $p_{\theta}(x|z)$.
4.  **Loss Computation**: Two terms are considered - reconstruction loss (how well the VAE reconstructs the data) and the KL divergence between $q_{\phi}(z|x)$ and $p(z)$.
5.  **Backpropagation and Optimization**: The model parameters $\phi$ and $\theta$ are updated to maximize the ELBO.

By the end of the training, you'll have a model that can generate new samples resembling your input data by simply sampling from the latent space and decoding the samples.

VAEs are a powerful tools, that stay in the intersection of deep learning and probabilistic modeling, and they have a plethora of applications, especially in unsupervised learning tasks.

## Variational Encoders with Pytorch

Let create a basic implementation of a Variational Autoencoder (VAE) using PyTorch. The VAE will be designed to work on simple image data, such as the MNIST dataset.

``` python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.latent_dim = latent_dim   # Add this line

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Loss function: Reconstruction + KL Divergence Losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 28, 28)[:n]])
                torchvision.utils.save_image(comparison.cpu(), 'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Run the training loop
epochs = 10
for epoch in range(1, epochs + 1):
    train(epoch)
    test()
```

```         
Train Epoch: 1 [0/60000 (0%)]   Loss: 547.095459
Train Epoch: 1 [12800/60000 (21%)]  Loss: 177.320297
Train Epoch: 1 [25600/60000 (43%)]  Loss: 156.426804
Train Epoch: 1 [38400/60000 (64%)]  Loss: 137.500916
Train Epoch: 1 [51200/60000 (85%)]  Loss: 130.676682
====> Epoch: 1 Average loss: 164.3802
====> Test set loss: 127.3049
Train Epoch: 2 [0/60000 (0%)]   Loss: 129.183395
Train Epoch: 2 [12800/60000 (21%)]  Loss: 124.367867
Train Epoch: 2 [25600/60000 (43%)]  Loss: 119.659966
Train Epoch: 2 [38400/60000 (64%)]  Loss: 120.912560
Train Epoch: 2 [51200/60000 (85%)]  Loss: 114.011864
====> Epoch: 2 Average loss: 121.6398
====> Test set loss: 115.7936
Train Epoch: 3 [0/60000 (0%)]   Loss: 114.913048
Train Epoch: 3 [12800/60000 (21%)]  Loss: 117.442482
Train Epoch: 3 [25600/60000 (43%)]  Loss: 111.994392
Train Epoch: 3 [38400/60000 (64%)]  Loss: 112.240242
Train Epoch: 3 [51200/60000 (85%)]  Loss: 114.725128
====> Epoch: 3 Average loss: 114.6564
====> Test set loss: 112.2248
Train Epoch: 4 [0/60000 (0%)]   Loss: 110.638550
Train Epoch: 4 [12800/60000 (21%)]  Loss: 114.595108
Train Epoch: 4 [25600/60000 (43%)]  Loss: 109.188904
Train Epoch: 4 [38400/60000 (64%)]  Loss: 111.060234
Train Epoch: 4 [51200/60000 (85%)]  Loss: 114.594086
====> Epoch: 4 Average loss: 111.6810
====> Test set loss: 109.6389
Train Epoch: 5 [0/60000 (0%)]   Loss: 110.394012
Train Epoch: 5 [12800/60000 (21%)]  Loss: 106.082031
Train Epoch: 5 [25600/60000 (43%)]  Loss: 107.659363
Train Epoch: 5 [38400/60000 (64%)]  Loss: 107.294495
Train Epoch: 5 [51200/60000 (85%)]  Loss: 110.049332
====> Epoch: 5 Average loss: 109.9291
====> Test set loss: 108.5438
Train Epoch: 6 [0/60000 (0%)]   Loss: 106.701828
Train Epoch: 6 [12800/60000 (21%)]  Loss: 109.286430
Train Epoch: 6 [25600/60000 (43%)]  Loss: 110.426498
Train Epoch: 6 [38400/60000 (64%)]  Loss: 106.086746
Train Epoch: 6 [51200/60000 (85%)]  Loss: 106.020401
====> Epoch: 6 Average loss: 108.7957
====> Test set loss: 107.6961
Train Epoch: 7 [0/60000 (0%)]   Loss: 109.973251
Train Epoch: 7 [12800/60000 (21%)]  Loss: 108.430046
Train Epoch: 7 [25600/60000 (43%)]  Loss: 109.439484
Train Epoch: 7 [38400/60000 (64%)]  Loss: 110.635895
Train Epoch: 7 [51200/60000 (85%)]  Loss: 110.213860
====> Epoch: 7 Average loss: 107.9552
====> Test set loss: 107.0711
Train Epoch: 8 [0/60000 (0%)]   Loss: 108.046188
Train Epoch: 8 [12800/60000 (21%)]  Loss: 105.081818
Train Epoch: 8 [25600/60000 (43%)]  Loss: 106.430084
Train Epoch: 8 [38400/60000 (64%)]  Loss: 106.380074
Train Epoch: 8 [51200/60000 (85%)]  Loss: 103.021561
====> Epoch: 8 Average loss: 107.3205
====> Test set loss: 106.6568
Train Epoch: 9 [0/60000 (0%)]   Loss: 106.435928
Train Epoch: 9 [12800/60000 (21%)]  Loss: 105.544891
Train Epoch: 9 [25600/60000 (43%)]  Loss: 102.952591
Train Epoch: 9 [38400/60000 (64%)]  Loss: 103.070465
Train Epoch: 9 [51200/60000 (85%)]  Loss: 105.689209
====> Epoch: 9 Average loss: 106.7969
====> Test set loss: 106.0421
Train Epoch: 10 [0/60000 (0%)]  Loss: 106.396545
Train Epoch: 10 [12800/60000 (21%)] Loss: 105.038795
Train Epoch: 10 [25600/60000 (43%)] Loss: 105.274765
Train Epoch: 10 [38400/60000 (64%)] Loss: 104.411789
Train Epoch: 10 [51200/60000 (85%)] Loss: 104.329590
====> Epoch: 10 Average loss: 106.3689
====> Test set loss: 105.5585
```

{{< admonition type=note title="Bibliography" open=false >}}
## Bibliography
- Doersch, Carl. 2021. “Tutorial on Variational Autoencoders.” January 3, 2021. http://arxiv.org/abs/1606.05908.
- Kingma, Diederik P., and Max Welling. 2019. “An Introduction to Variational Autoencoders.” Foundations and Trends® in Machine Learning 12 (4): 307–92. https://doi.org/10.1561/2200000056.
- Ramchandran, Siddharth, Gleb Tikhonov, Otto Lönnroth, Pekka Tiikkainen, and Harri Lähdesmäki. 2022. “Learning Conditional Variational Autoencoders with Missing Covariates.” March 2, 2022. http://arxiv.org/abs/2203.01218.
- Yunfan Jiang, ELBO — What & Why,Jan 11, 2021, in https://yunfanj.com/blog/2021/01/11/ELBO.html.
{{< /admonition >}}

Pic by <a href="https://www.freepik.es/foto-gratis/laboratorio-computacion-brillante-equipo-moderno-tecnologia-generada-ia_41451597.htm">@vecstock</a>, <a href="https://www.freepik.es/">Freepik</a>