# Denoising Diffusion Probabilistic Models (DDPM)

Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn to generate data by reversing a gradual noising process. The model is trained to denoise data step by step, starting from pure noise and iteratively refining it to produce realistic samples.

## Problem Formulation

We are given a dataset of samples $\mathbf{X}_0 = \{x_1, x_2, ..., x_N\}$, where each $x_i$ is a high-dimensional data sample.

The true form of the underlying distribution is unknown, but we can assume that it can be represented by a probability density function $p_{data}(\mathbf{X}_0)$. The subscript $0$ indicates that this is the original, uncorrupted data.

Let's introduce latent variables of the same dimensionality as the data defined as $ \mathbf{X}_{1:T} = \{\mathbf{X}_1, \mathbf{X}_2, ..., \mathbf{X}_T\} $. Then the joint distribution over the observed data and the latent variables can be written as $p_\phi(\mathbf{X}_{0:T})$.

DDPM's model the data distribution as the marginal over the joint distribution of the observed data and the latent variables:

$$
p_\phi(\mathbf{X}_0) = \int p(\mathbf{X}_{0:T}) \, d\mathbf{X}_{1:T}
$$

where $\mathbf{X}_{1:T} = \{\mathbf{X}_1, \mathbf{X}_2, ..., \mathbf{X}_T\}$ represents a sequence of latent variables. 

**Markov Chain Assumption**: The joint distribution over the observed data and the latent variables can be factorized as a Markov chain:

$$
p_\phi(\mathbf{X}_{0:T}) = p_\phi(\mathbf{X}_T) \prod_{t=1}^{T} p_\phi(\mathbf{X}_{t-1} | \mathbf{X}_t)
$$

The Markov chain assumption in DDPM's was inspired by a similar assumption made in non-equilibrium statistical physics and 
sequential monte carlo methods. TODO: Verify! And come back to Markov Chain modeling.

## Maximum Likelihood Estimation

The log-likelihood of the observed data under the model is given by:

$$
\log p_\phi(\mathbf{X}_0) = \log \int p_\phi(\mathbf{X}_{0:T}) \, d\mathbf{X}_{1:T}
$$

Using the Markov chain factorization of the joint distribution, we can rewrite this as:

$$
\log p_\phi(\mathbf{X}_0) = \log \int p_\phi(\mathbf{X}_T) \prod_{t=1}^{T} p_\phi(\mathbf{X}_{t-1} | \mathbf{X}_t) \, d\mathbf{X}_{1:T}
$$

Since the integral is intractable, we typically resort to variational inference or other approximation methods to optimize the log-likelihood. In the context of DDPMs, this leads to the derivation of a variational lower bound (VLB) on the log-likelihood, which can be efficiently optimized.

Let's derive the VLB. 

We introduce a variational distribution $q(\mathbf{X}_{1:T}|\mathbf{X}_0)$ to approximate the true posterior over the latent variables. 
We multiply and divide the joint distribution by the variational distribution inside the logarithm to obtain:

$$
\log p_\phi(\mathbf{X}_0) = \log \int q(\mathbf{X}_{1:T}|\mathbf{X}_0) \frac{p_\phi(\mathbf{X}_{0:T})}{q(\mathbf{X}_{1:T}|\mathbf{X}_0)} \, d\mathbf{X}_{1:T}
$$


Applying Jensen's inequality, we get the variational lower bound:

$$
\log p_\phi(\mathbf{X}_0) \geq \mathbb{E}_{q(\mathbf{X}_{1:T}|\mathbf{X}_0)} \left[ \log \frac{p_\phi(\mathbf{X}_{0:T})}{q(\mathbf{X}_{1:T}|\mathbf{X}_0)} \right]
$$
