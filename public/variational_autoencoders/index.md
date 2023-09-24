# Generative models: variational autoencoders


Generative models are a class of statistical models that aim to learn the underlying data distribution from a given dataset. These models provide a way to generate new samples that are statistically similar to the training data. They have gained substantial attention in various domains, such as image generation, speech synthesis, and even drug discovery.

<!--more-->

### Generative Model

Generative models are a class of statistical models that aim to learn the underlying data distribution. Given a dataset of observed samples, one starts by selecting a distributional model parameterized by $(\theta)$. The objective is to estimate $(\theta)$ such that it aligns optimally with the observed samples.The anticipation is that it can also generalize to samples outside the training set.

The optimal distribution is hence the one that maximizes the likelihood of producing the observed data, giving lower probabilities to infrequent observations and higher probabilities to the more common ones (the principle underlying this assumption is: 'the world is a boring place' Bhiksha Raj).

### The Challenge of Maximum Likelihood Estimates (MLE) for Unseen Observations

When training generative models, a natural objective is to optimize the model parameters such that the likelihood of the observed data under the model is maximized. This method is known as **Maximum Likelihood Estimation (MLE)**. In mathematical terms, given observed data $X$, the MLE seeks parameters $\theta$ that maximize:

> $p_\theta(X)$

However, for many generative models, especially those that involve latent or unobserved variables, the likelihood term involves summing or integrating over all possible configurations of these latent variables. Mathematically, this turns into:

> $p_\theta(X) = \sum_{Z} p_\theta(X,Z)$\
> or\
> $p_\theta(X) = \int p_\theta(X,Z) dZ$

Computing the log-likelihood, which is often used for numerical stability and optimization ease, leads to a log of summations (for discrete latent variables) or a log of integrals (for continuous latent variables):

> $log p_\theta(X) = \log \sum_{Z} p_\theta(X,Z)$\
> or\
> $log p_\theta(X) = \log \int p_\theta(X,Z) dZ$

These expressions are typically intractable to optimize directly due to the presence of the log-sum or log-integral operations. Where do these formulas come frome? They come forom *marginalization* in the context of joint probability.

### Marginalization in the Context of Joint Probability

When discussing the computation of the joint probability for observed and missing data, the term "marginalizing" refers to summing or integrating over all possible outcomes of the missing data. This process provides a probability distribution based solely on the observed data. For example, let's assume:

-   $X$ is the observed data
-   $Z$ is the missing data
-   The joint probability for both is represented as $p(X,Z)$

If your primary interest lies in the distribution of $X$ and you wish to eliminate the dependence on $Z$, you'll need to carry out marginalization for $Z$. For discrete variables, the marginalization involves the logarithm of summation:

>$$ \log \left( \sum_{z} p(X,Z=z) \right),$$

for continuous variables, it pertains to integration:

>$$ \int p(X,Z) dZ $$

However, it's essential to note that these functions includes the log of a sum, which defies direct optimization. Can we get an approximation to this that is more tractable (without a summation or integral within the log)?

### Overcoming the Challenge with Expectation Maximization (EM)

To address the optimization challenge in MLE with latent variables, the **Expectation Maximization (EM)** algorithm is employed. The EM algorithm offers a systematic approach to iteratively estimate both the model parameters and the latent variables.

The algorithm involves two main steps:

1.  **E-step (Expectation step)**: involves computing the expected value of the complete-data log-likelihood with respect to the posterior distribution of the latent variables given the observed data.
2.  **M-step (Maximization step)**: Update the model parameters to maximize this expected log-likelihood from the E-step.

By alternating between these two steps, EM ensures that the likelihood increases with each iteration until convergence, thus providing a practical method to fit generative models with latent variables.

For E-step the **Variational Lower Bound** is used. Commonly referred to as the Empirical Lower BOund (ELBO), is a central concept in variational inference. This method is used to approximate complex distributions (typically posterior distributions) with simpler, more tractable ones. The ELBO is an auxiliary function that models MLE by iteratibly maximize a 'lower bound' function

Variational inference is set up as an optimization problem where the objective is to find a distribution $q$ from a simpler family of distributions that is close to the target distribution $p$. The closeness is measured using the Kullback-Leibler (KL) divergence. The ELBO serves as the objective function to be maximized in this optimization.

The ELBO can be derived from the logarithm of the marginal likelihood of the observed data. Given:

> $p(\mathbf{x})$ is the marginal likelihood of the observed data $\mathbf{x}$,
>
> $q(\mathbf{z})$ is the variational distribution over the latent variables $\mathbf{z}$,
>
> $p(\mathbf{z} | \mathbf{x})$ is the true posterior distribution of $\mathbf{z}$.

The ELBO is given by:

>${ELBO}(q) = \mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_q[\log q(\mathbf{z})]$

Where:

$\mathbb{E}_q$ denotes the expectation with respect to the distribution $q$. The first term on the right-hand side measures how well the joint distribution $p(\mathbf{x}, \mathbf{z})$ is modeled by $q$. -The second term is the entropy of $q$, which acts as a regularizer. Maximizing the ELBO is equivalent to minimizing the KL divergence between the variational distribution $q$ and the true posterior $p(\mathbf{z} | \mathbf{x})$. Intuitively, by maximizing the ELBO, you're trying to find a balance between a distribution $q$ that closely matches the target distribution $p$ (as measured by the joint likelihood) and one that maintains uncertainty (as measured by its entropy).

In SUMMARY, generative models offer a powerful approach to understanding and generating data. The challenges posed by the MLE in the presence of latent variables are effectively addressed by the EM algorithm, making it a cornerstone method in the training of many generative models.

### Variational Autoencoders (VAEs)

Variational Autoencoders are a specific type of generative model that brings together ideas from deep learning and Bayesian inference. VAEs are especially known for their application in generating new, similar data to the input data (like images or texts) and for their ability to learn latent representations of data.


**1. Generative Models and Latent Variables**

In generative modeling, our goal is to learn a model of the probability distribution from which a dataset is drawn. The model can then be used to generate new samples. A VAE makes a specific assumption that there exist some *latent variables* (or hidden variables) that when transformed give rise to the observed data.

Let $x$ be the observed data and $z$ be the latent variables. The generative story can be seen as:

1. Draw $z$ from a prior distribution, $p(z)$.
2. Draw $x$ from a conditional distribution, $p(x|z)$.

**2. Problem of Direct Inference**

As discussed previously, direct inference for the posterior distribution $p(z|x)$ (i.e., the probability of the latent variables given the observed data) can be computationally challenging, especially when dealing with high-dimensional data or complex models. This is because:

$$ p(z|x) = \frac{p(x|z) p(z)}{p(x)} $$

Here, $p(x)$ is the evidence (or marginal likelihood) which is calculated as:

$$ p(x) = \int p(x|z) p(z) dz $$

This integral is intractable for most interesting models.

**3. Variational Inference and ELBO**

To sidestep the intractability of the posterior, VAEs employ *variational inference*. Instead of computing the posterior directly, we introduce a parametric approximate posterior distribution, $q_{\phi}(z|x)$, with its own parameters $\phi$. 

The goal now shifts to making this approximation as close as possible to the true posterior. This is done by minimizing the *Kullback-Leibler divergence* between the approximate and true posterior. 

The optimization objective, known as the **Evidence Lower BOund (ELBO)**, can be written as:    

>$$ \text{ELBO}(\phi) = \mathbb{E}_{q_{\phi}(z|x)}[\log p(x|z)] - \text{KL}(q_{\phi}(z|x) || p(z)) $$

Where KL represents the Kullback-Leibler divergence.

**4. Neural Networks and Autoencoding Structure**

In VAEs, neural networks are employed to parameterize the complex functions. Specifically:

1. **Encoder Network**: This maps the observed data, $x$, to the parameters of the approximate posterior, $q_{\phi}(z|x)$.
2. **Decoder Network**: Given samples of $z$ drawn from $q_{\phi}(z|x)$, this maps back to the data space, outputting parameters for the data likelihood, $p_{\theta}(x|z)$.

The "autoencoder" terminology comes from the encoder-decoder structure where the model is trained to reconstruct its input data.

**5. Training a VAE**

The training process involves:

1. **Forward pass**: Input data is passed through the encoder to obtain parameters of $q_{\phi}(z|x)$.
2. **Sampling**: Latent variables $z$ are sampled from $q_{\phi}(z|x)$ using the *reparameterization trick* for backpropagation.
3. **Reconstruction**: The sampled $z$ values are passed through the decoder to obtain the data likelihood parameters, $p_{\theta}(x|z)$.
4. **Loss Computation**: Two terms are considered - reconstruction loss (how well the VAE reconstructs the data) and the KL divergence between $q_{\phi}(z|x)$ and $p(z)$.
5. **Backpropagation and Optimization**: The model parameters $\phi$ and $\theta$ are updated to maximize the ELBO.

By the end of the training, you'll have a model that can generate new samples resembling your input data by simply sampling from the latent space and decoding the samples.

VAEs are a powerful tools, that stay in the intersection of deep learning and probabilistic modeling, and they have a plethora of applications, especially in unsupervised learning tasks.

### Variational Encoders with Pytorch

Let create a basic implementation of a Variational Autoencoder (VAE) using PyTorch. The VAE will be designed to work on simple image data, such as the MNIST dataset.

```python
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
