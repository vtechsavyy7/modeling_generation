# Denoising Diffusion Probabilistic Models (DDPM)

Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn to generate data by reversing a gradual noising process. The model is trained to denoise data step by step, starting from pure noise and iteratively refining it to produce realistic samples.

## Resources: 

- [Denoising Diffusion Probabilistic Models (DDPM) Paper](https://arxiv.org/abs/2006.11239)
- [Ritwik Raha's Blog on DDPM](https://blog.ritwikraha.dev/ddpm-explained-for-dummies)
- [AI Summer article on Diffusion Models](https://theaisummer.com/diffusion-models/)

## Problem Formulation

We are given a dataset of samples denoted as $\boldsymbol{x}_0 = \{x^1, x^2, ..., x^N\}$, where each $x^i$ is a high-dimensional data sample.

The true form of the underlying distribution is unknown, but we can assume that it can be represented by a probability density function $p_{data}(\boldsymbol{x}_0)$. The subscript $0$ indicates that this is the original, uncorrupted data.

Let's introduce latent variables of the same dimensionality as the data defined as $\boldsymbol{x}_{1:T} = \{\boldsymbol{x}_1, \boldsymbol{x}_2, ..., \boldsymbol{x}_T\}$. Then the joint distribution over the observed data and the latent variables can be written as $p_\phi(\boldsymbol{x}_{0:T})$, where $\phi$ represents the parameters of the distribution that we want to learn.

DDPM's model the original data distribution as the marginal over the joint distribution of the observed data and the latent variables:

$$
p_\phi(\boldsymbol{x}_{0:T}) \, d\boldsymbol{x}_{1:T}
$$

where $\boldsymbol{x}_{1:T} = \{\boldsymbol{x}_1, \boldsymbol{x}_2, ..., \boldsymbol{x}_T\}$ are the latent variables. 

**Markov Chain Assumption**: Diffusion models make the assumption that the joint distribution over the observed data and the latent variables can be factorized as a Markov chain:

$$
p_\phi(\boldsymbol{x}_{0:T}) = p(\boldsymbol{x}_T) \prod_{t=1}^{T} p_\phi(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t)
$$

The Markov Chain starts from a prior distribution over the final latent variable $\boldsymbol{x}_T$, which is chosen to be a standard Gaussian:

$$
p(\boldsymbol{x}_T) = \mathcal{N}(\boldsymbol{x}_T; \boldsymbol{0}, \boldsymbol{I})
$$

The Markov chain assumption in DDPM's was inspired by a similar assumption made in non-equilibrium statistical physics and 
sequential monte carlo methods.

A further modeling assumption that is made is that the conditional distrbutions $p_\phi(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t)$ are Gaussian distributions with mean and variance that are functions of $\boldsymbol{x}_t$ and the time step $t$:

$$
p_\phi(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t) = \mathcal{N}(\boldsymbol{x}_{t-1}; \mu_\phi(\boldsymbol{x}_t, t), \Sigma_\phi(\boldsymbol{x}_t, t))
$$

## Maximum Likelihood Estimation

The log-likelihood of the original, uncorrupted data under the model is given by:

$$
\log p_\phi(\boldsymbol{x}_0) = \log \int p_\phi(\boldsymbol{x}_{0:T}) \, d\boldsymbol{x}_{1:T}
$$

Using the Markov chain factorization of the joint distribution, we can rewrite this as:

$$
\log p_\phi(\boldsymbol{x}_0) = \log \int p_\phi(\boldsymbol{x}_T) \prod_{t=1}^{T} p_\phi(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t) \, d\boldsymbol{x}_{1:T}
$$

Since the integral is intractable, we typically resort to variational inference or other approximation methods to optimize the log-likelihood. 

In the context of DDPMs, we introduce a variational distribution $q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)$ to approximate the true posterior over the latent variables.

We multiply and divide the joint distribution by the variational distribution inside the logarithm to obtain:

$$
\log p_\phi(\boldsymbol{x}_0) = \log \int q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) \frac{p_\phi(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \, d\boldsymbol{x}_{1:T}
$$

Applying Jensen's inequality, we get the variational lower bound $\mathcal{L}$:

$$
\mathcal{L} = \mathbb{E}_{q} \left[ \log \frac{p_\phi(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \right]
$$

Note: For notational convenience, we often abbreviate $q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)$ as $q$ in the expectation.

## Forward Diffusion Process

One of the key distinguishing features of DDPMs compared to other latent variable models is that the approximate posterior $q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)$ is defined as a fixed (non-learned) Markov chain of Gaussian transitions:

$$
q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1}) = \mathcal{N}(\boldsymbol{x}_t; \sqrt{1-\beta_t} \boldsymbol{x}_{t-1}, \beta_t \boldsymbol{I})
$$

where $\beta_t$ is a variance schedule that controls the amount of noise added at each time step. This Markov chain is called the forward diffusion process and is fixed, meaning it is not learned during training.

### TODO: Variance preserving property of the forward process


## Simplifying the Variational Lower Bound

Given this factorization, we can rewrite the variational lower bound as a sum of terms corresponding to each time step:

$$
\mathcal{L} = \mathbb{E}_{q} \left[ \log \frac{p_\phi(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \right] = \mathbb{E}_{q} \left[ \log p(\boldsymbol{x}_T) + \log \prod_{t=1}^{T} \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})} \right]
$$

One further expansion that we make is: 

$$
\mathcal{L} = \mathbb{E}_{q} \left[ \log  p(\boldsymbol{x}_T) + \log \left( \prod_{t=2}^{T} \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})} \cdot \frac{p_\phi(\boldsymbol{x}_0|\boldsymbol{x}_1)}{q(\boldsymbol{x}_1|\boldsymbol{x}_0)} \right) \right]
$$

Let's focus on just the products in the second term of the expectation. Let's denote it as $\mathcal{P}$

$$
\mathcal{P} = \prod_{t=2}^{T} \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})}
$$

The forward transitions $q(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$ cannot be directly compared to the reverse transitions $p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$ because the conditioning is flipped: the forward process conditions on the previous state, while the reverse process conditions on the next state.

### Conditioning Trick: 

A central idea of DDPM's is to condition the forward transitions on the original data $\boldsymbol{x}_0$, and use Bayes rule to express the reverse transitions in terms of the forward transitions:

$$
q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})  = q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1}, \boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0) q(\boldsymbol{x}_{t}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)}
$$

This allows us to relate the reverse process to the known forward process, facilitating the derivation of the variational lower bound.

<u>Note</u>: The equality $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$ holds because the forward process is a Markov chain conditioned on $\boldsymbol{x}_0$, meaning that $\boldsymbol{x}_{t-1}$ is independent of $\boldsymbol{x}_t$ given $\boldsymbol{x}_0$. This is a key property that allows us to express the reverse transitions in terms of the forward transitions and the original data.

Substituting the expression for $q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1})$ from above into $\mathcal{P}$, we get:

$$
\mathcal{P} =  \prod_{t=2}^{T} \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)} \cdot \frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_{t}|\boldsymbol{x}_0)} 
$$

If we expand out the terms in the product it's easy to see that $ q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)$ in the numerator of one term cancels with $ q(\boldsymbol{x}_{t}|\boldsymbol{x}_0)$ in the denominator of the next term, leaving only the first and last terms:
$$
\prod_{t=2}^{T} \frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_{t}|\boldsymbol{x}_0)} = \frac{q(\boldsymbol{x}_{1}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_{T}|\boldsymbol{x}_0)} 
$$

Thus, 

$$
\mathcal{P} = \prod_{t=2}^{T} \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)} \cdot \frac{q(\boldsymbol{x}_{1}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_{T}|\boldsymbol{x}_0)}
$$

Substituting $\mathcal{P}$ back into the expression for $\mathcal{L}$ we get:

$$
\mathcal{L} = \mathbb{E}_{q} \left[ \log  p(\boldsymbol{x}_T) + \log \left( \prod_{t=2}^{T} \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)} \cdot \frac{q(\boldsymbol{x}_{1}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_{T}|\boldsymbol{x}_0)} \cdot \frac{p(\boldsymbol{x}_0|\boldsymbol{x}_1)}{q(\boldsymbol{x}_1|\boldsymbol{x}_0)} \right) \right]
$$

Notice that the term $q(\boldsymbol{x}_1|\boldsymbol{x}_0)$ cancels out, simplifying the expression for $\mathcal{L}$.

$$
\mathcal{L} = \mathbb{E}_{q} \left[ \log  p(\boldsymbol{x}_T) + \log \left( \prod_{t=2}^{T} \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)} \cdot \frac{p(\boldsymbol{x}_0|\boldsymbol{x}_1)}{q(\boldsymbol{x}_T|\boldsymbol{x}_0)} \right) \right]
$$

Now taking the logarithm of the product, we can rewrite it as a sum of logarithms:

$$
\mathcal{L} = \mathbb{E}_{q} \left[ \log  p(\boldsymbol{x}_T) + \sum_{t=2}^{T} \log \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)} + \log p_\phi(\boldsymbol{x}_0|\boldsymbol{x}_1) - \log q(\boldsymbol{x}_T|\boldsymbol{x}_0) \right]
$$

The first and the fourth term can be combined into a single logarithm, to give a more compact expression

$$
\mathcal{L} = \mathbb{E}_{q} \left[ \sum_{t=2}^{T} \log \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)} + \log p_\phi(\boldsymbol{x}_0|\boldsymbol{x}_1) + \log \frac{p(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T|\boldsymbol{x}_0)} \right]
$$

Taking the individual expectations, we can rewrite $\mathcal{L}$ as 

$$
\mathcal{L} = \mathbb{E}_{q} \left[ \sum_{t=2}^{T} \log \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)} \right] + \mathbb{E}_{q} \left[ \log p_\phi(\boldsymbol{x}_0|\boldsymbol{x}_1) \right] + \mathbb{E}_{q} \left[ \log \frac{p(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T|\boldsymbol{x}_0)} \right]
$$

## Decomposition of the Variational Lower Bound: 

Looking at the individual terms of the variational lower bound $\mathcal{L}$

- Prior Matching Term ($\mathcal{L}_T$)

$$
\mathcal{L}_T = \mathbb{E}_{q} \left[ \log \frac{p(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T|\boldsymbol{x}_0)} \right]  = -\mathbb{E}_{q} \left[ \log \frac{q(\boldsymbol{x}_T|\boldsymbol{x}_0)}{p(\boldsymbol{x}_T)} \right] = - \text{KL}\left[ q(\boldsymbol{x}_T|\boldsymbol{x}_0) \| p(\boldsymbol{x}_T) \right]
$$

By definition of the KL divergence, this term measures how well the prior $p(\boldsymbol{x}_T)$ matches the distribution of the final latent variable $q(\boldsymbol{x}_T|\boldsymbol{x}_0)$. Since the forward diffusion process is fixed and not learnable, this term of the lower bound is also fixed and does not depend on the learnable parameters of the model. So it can be excluded from the loss computation during training.

- Reconstruction Term ($\mathcal{L}_0$)

$$
\mathcal{L}_0 = \mathbb{E}_{q} \left[ \log p_\phi(\boldsymbol{x}_0|\boldsymbol{x}_1) \right]
$$

This term measures how well the model can reconstruct the original data $\boldsymbol{x}_0$ from the first latent variable $\boldsymbol{x}_1$. Unlike the prior matching term, this term depends on the learnable parameters of the model and is crucial for training the reverse diffusion process.

- Intermediate KL Terms ($\mathcal{L}_{t-1}$)

For $t = 2, \dots, T$, we have the intermediate KL terms:

$$
\mathcal{L}_{t-1} = \mathbb{E}_{q} \left[ \log \frac{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)}{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)} \right] = - \mathbb{E}_{q} \left[ \log \frac{q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)}{p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)} \right] = - \text{KL}\left[ q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0) \| p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) \right]
$$

These terms measure how well the model's reverse process $p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$ approximates the true posterior $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)$ at each intermediate step. Unlike the prior matching term, these terms depend on the learnable parameters and are essential for training the reverse diffusion process.

The variational lower bound needs to be maximized with respect to the learnable parameters of the model. In practice, this is equivalent to minimizing the negative variational lower bound, which can be expressed as the sum of the negative reconstruction term and the negative intermediate KL terms, while the prior matching term can be ignored during training since it is fixed.

Hence, the loss function used for training the model can be written as:

$$
\mathcal{L}_{\text{train}} = -\mathcal{L}_0 - \sum_{t=2}^{T} \mathcal{L}_{t-1} = \mathbb{E}_{q} \left[ -\log p_\phi(\boldsymbol{x}_0|\boldsymbol{x}_1) + \sum_{t=2}^{T} \text{KL}\left[ q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0) \| p_\phi(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) \right] \right]
$$

This formulation allows the model to focus on learning the reverse diffusion process effectively, without being influenced by the fixed prior matching term.


## Intermediate KL Divergences

### Estimating the Posterior $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)$

We still have a problem of intractability when it comes to computing the intermediate KL divergence terms, as they involve an expectation over the true posterior $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)$, which is unknown.

However, from equation xx, we can re-wrtie the posterior as

$$
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_{t}|\boldsymbol{x}_{t-1}) \cdot q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_0)}{q(\boldsymbol{x}_{t} | \boldsymbol{x}_0)}
$$

All three terms in the RHS are known from the fixed forward diffuusion process. All 3 of them are Gaussian distributions, which allows us to compute the posterior $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0)$ in closed form as another Gaussian distribution given by: 

$$
q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_{t}, \boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0), \tilde{\beta}_t \mathbf{I})
$$

where

$$
\tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{\sqrt{\alpha_t} (1 - \alpha_{t-1})}{1 - \alpha_t} \boldsymbol{x}_0 + \frac{\sqrt{\alpha_{t-1}} \beta_t}{1 - \alpha_t} \boldsymbol{x}_t
$$
and
$$
\tilde{\beta}_t = \frac{(1 - \alpha_{t-1}) \beta_t}{1 - \alpha_t}
$$

The derivation of this Gaussian distribution is presented in Appendix A (TODO)


### Parameterizing the Reverse Process

In the DDPM paper, the reverse process is parameterized as a Gaussian distribution with mean and variance predicted by a neural network. Specifically, the reverse process is defined as:

$$
p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t) = \mathcal{N}(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t), \Sigma_\theta(\boldsymbol{x}_t, t))
$$

where $\boldsymbol{\mu}_\theta$ and $\Sigma_\theta$ are the mean and covariance predicted by the neural network with parameters $\theta$. The network is trained to approximate the true posterior $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$ derived above.

For simplification, the authors choose to nor learn the covaraince matrix of the reverse transition and instead fix it to a time-dependent constant $\Sigma_\theta(\boldsymbol{x}_t, t) = \sigma_t^2 \mathbf{I}$.


### Closed-form KL Divergence

Given that both the true posterior $q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0)$ and the reverse process $p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)$ are Gaussian distributions, the KL divergence between them can be computed in closed form. For two multivariate and isotropic Gaussian distributions $\mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$ and $\mathcal{N}(\boldsymbol{\mu}_2, \sigma_2^2 \mathbf{I})$, the KL divergence is given by:

$$
\text{KL}(\mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I}) \| \mathcal{N}(\boldsymbol{\mu}_2, \sigma_2^2 \mathbf{I})) = \frac{1}{2} \left[ d \log \frac{\sigma_2^2}{\sigma_1^2} - d + \frac{d \sigma_1^2}{\sigma_2^2} + \frac{\|\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1\|^2}{\sigma_2^2} \right]
$$

where $d$ is the dimensionality of the Gaussian distributions. This formula can be directly applied to compute $\text{KL}(q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) \| p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t))$.

$$
\text{KL}(q(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t, \boldsymbol{x}_0) \| p_\theta(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t)) = \frac{1}{2} \left[ d \log \frac{\sigma_t^2}{\tilde{\beta}_t} - d + \frac{d \tilde{\beta}_t}{\sigma_t^2} + \frac{\|\boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t) - \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0)\|^2}{\sigma_t^2} \right]
$$

The first 3 terms involving the variances do not depend on the model parameters $\theta$ and thus can be ignored during training. Therefore, the training objective can be simplified to minimizing the squared error between the predicted mean and the true posterior mean:

$$
L_t =  \mathbb{E}_q \left[ \frac{1}{2 \sigma_t^2} \|\boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t) - \tilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0)\|^2 \right] + C
$$


### Reparameterize to Predict Noise (TODO)


### Training Objective (TODO)

### Simplified Training Objective (TODO)