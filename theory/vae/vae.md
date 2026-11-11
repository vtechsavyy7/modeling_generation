# Variational Autoencoders (VAEs)

## Resources

- [Kingma and Welling's Original Paper](https://arxiv.org/abs/1312.6114)
- [Matthew Bernstein's Article](https://mbernste.github.io/posts/vae/)


## Problem Formulation

Let $\bold{X} = \{x_1, x_2, ..., x_n\}$ be a set of observed data points, where each $x_i$ is a high-dimensional vector.  </br>

The goal of Variational Autoencoders (VAEs) is to learn a probabilistic model that can generate new data points similar to the observed data.
i.e we want to model the underlying distribution of the data and be able to sample from it. 

Let us denote the probabilistic model as $p_\theta(\mathbb{X})$, where $\theta$ represents the parameters of the model.

TODO (Verify): Explain why we are going with a latent variable model for the data.

The reason for introducing latent variables is that high-dimensional data often lies on a lower-dimensional manifold. 
By modeling the data with latent variables, we can capture the underlying structure and dependencies in the data more effectively.
The latent variable $z$ serves as a compact representation of the data, allowing the model to generate new samples by first sampling $z$ from a prior distribution 
and then mapping it to the data space through the generative model $p_\theta(x|z)$.

### Evidence Lower Bound (ELBO)

- [Matthew Bernstein's Article](https://mbernste.github.io/posts/elbo/)

- The Evidence Lower Bound (ELBO) is a key concept in VAEs. It provides a tractable lower bound on the log-likelihood of the observed data, which we can maximize to learn the model parameters.

- Let's start by defining the log-likelihood of the observed data
$$
\log p(\mathbb{X}) = \log p(x_1, x_2, ..., x_n)
$$

- Let's also introduce a variational distribution $q_\phi(z|x)$ to approximate the true posterior $p(z|x)$, where $z$ is a latent variable and $\phi$ represents the parameters of the approximate posterior.

- Now let's expand the log-likelihood of a single data point $x$ using the variational distribution $q_\phi(z|x)$:

$$
\log p(x) = \log \int p(x, z) \, dz = \log \int q_\phi(z|x) \frac{p(x, z)}{q_\phi(z|x)} \, dz
$$

- Applying Jensen's inequality, we obtain a lower bound on the log-likelihood, which is the Evidence Lower Bound (ELBO):

$$
\log p(x) \geq \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p(x, z)}{q_\phi(z|x)} \right]
$$


### ELBO Decomposition for VAE loss

- The ELBO can be further decomposed as:

$$
\begin{aligned}
ELBO &= \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p_\theta(x | z) p(z)}{q_\phi(z|x)} \right] \\
&= \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{q_\phi(z|x)}{p(z)} \right] \\
&= \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))
\end{aligned}
$$

- In practice, the VAE loss function is defined as the negative ELBO, which we aim to minimize:
$$
\mathcal{L}_{\text{VAE}} = - \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] + \text{KL}(q_\phi(z|x) \| p(z))
$$

- The first term is the reconstruction loss, encouraging the model to accurately reconstruct the input data from the latent representation.
- The second term is the KL divergence, acting as a regularizer to ensure that the approximate posterior $q_\phi(z|x)$ stays close to the prior $p(z)$. We typically choose a simple prior, such as a standard normal distribution, to make the KL divergence term analytically tractable.

### Regularization loss in VAE

- In the VAE loss equation, we still have to define the generative distribution $p_\theta(x|z)$ and the variational distribution $q_\phi(z|x)$. 
- Let's start by modeling $q_\phi(z|x)$, the variational distribution or the Encoder, as a Multivariate Gaussian distribution with a diagonal covariance matrix. 

$$
q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \Sigma_\phi(x))
$$

- The encoder neural network takes the input data $x$ and outputs two vectors: the mean vector $\mu_\phi(x)$ and the log-variance vector $\log \sigma^2_\phi(x)$ (which defines the diagonal elements of $\Sigma$). 

- Since $q_\phi(z|x)$ and $p(z)$ are both Gaussian distributions, the KL divergence term in the VAE loss can be computed analytically, which simplifies the optimization process. Since $p(z)$ is assumed to be a standard normal distribution $ \mathcal{N}(0, I)$, the KL divergence has a closed-form expression given by:

$$
\text{KL}(q_\phi(z|x) \| p(z)) = \frac{1}{2} \sum_i \left( \sigma^2_{\phi,i}(x) + \mu^2_{\phi,i}(x) - 1 - \log \sigma^2_{\phi,i}(x) \right)
$$

TODO: Later derive the closed-form expression for the KL divergence starting from the definition of the multivariate Gaussian distributions.

## Reconstruction loss in VAE

- The form chosen for the generative distribution $p_\theta(x|z)$ depends on the type of data being modeled. For continuous data, it is common to assume a Gaussian distribution with a mean given by the output of the decoder network and a fixed variance:

$$
p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)
$$

- Now given this form, how do we compute the reconstruction loss term in the VAE loss function?

- We combine a mathematical simplification with a sampling trick called the Monte Carlo approximation.

- Here is the exact step-by-step breakdown of how the expectation $\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]$ transforms into the familiar Mean Squared Error (MSE) loss used in code.

**Step 1**: Write Out the Log-Likelihood of the Gaussian

Suppose that $D$ is the dimensionality of the data vector $x$. Then the Gaussian likelihood can be written as:
$$
p_{\theta }(x\mid z)=\mathcal{N}(x;\,\mu _{\theta }(z),\,\sigma ^{2}I)=\left(\frac{1}{\sqrt{2\pi \sigma ^{2}}}\right)^{D}\exp \left(-\frac{\|{}x-\mu _{\theta }(z)\|{}^{2}}{2\sigma ^{2}}\right)
$$

Taking the natural logarithm ($\log $) of both sides converts the product/exponent into a sum:
$$
\log p_{\theta }(x\mid z)=\log \left[\left(2\pi \sigma ^{2}\right)^{-\frac{D}{2}}\right]-\frac{\|{}x-\mu _{\theta }(z)\|{}^{2}}{2\sigma ^{2}}
$$
$$
\log p_{\theta }(x\mid z)=-\frac{D}{2}\log (2\pi \sigma ^{2})-\frac{1}{2\sigma ^{2}}\sum _{i=1}^{D}(x_{i}-\mu _{\theta ,i}(z))^{2}
$$

**Step 2**: Handle the Expectation $\mathbb{E}_{q_{\phi }(z\mid x)}$

Now we plug this log-likelihood back into the expectation over the encoder's distribution $q_\phi(z \mid x)$:
$$
\mathbb{E}_{q_{\phi }(z\mid x)}[\log p_{\theta }(x\mid z)]=\mathbb{E}_{q_{\phi }(z\mid x)}\left[-\frac{D}{2}\log (2\pi \sigma ^{2})-\frac{1}{2\sigma ^{2}}\|{}x-\mu _{\theta }(z)\|{}^{2}\right]
$$
Because the first term ($-\frac{D}{2}\log(2\pi\sigma^2)$) and the scalar multiplier ($\frac{1}{2\sigma ^{2}}$) do not depend on $z$, they can be pulled completely out of the expectation:
$$
\mathbb{E}_{q_{\phi }(z\mid x)}[\log p_{\theta }(x\mid z)]=-\frac{D}{2}\log (2\pi \sigma ^{2})-\frac{1}{2\sigma ^{2}}\mathbb{E}_{q_{\phi }(z\mid x)}\left[\|{}x-\mu _{\theta }(z)\|{}^{2}\right]
$$

**Step 3**: Solve the Expectation via Monte Carlo Sampling

Evaluating the remaining expectation analytically is impossible because $\mu_\theta(z)$ is a complex, non-linear function wrapped inside a deep neural network (the decoder).
To solve this, we use a Monte Carlo approximation. We draw $L$ random samples of $z$ from our encoder distribution $q_\phi(z \mid x)$ and calculate the average:
$$
\mathbb{E}_{q_{\phi }(z\mid x)}\left[\|{}x-\mu _{\theta }(z)\|{}^{2}\right]\approx \frac{1}{L}\sum _{l=1}^{L}\|{}x-\mu _{\theta }(z^{(l)})\|{}^{2}\quad \text{where\ }z^{(l)}\sim q_{\phi }(z\mid x)
$$


In practice, when training VAEs with mini-batch stochastic gradient descent, setting $L = 1$ (drawing just one single sample of $z$ per data point per iteration) is completely sufficient. As long as your batch size is reasonably large (e.g., 32, 64, 128), the variations average out across the batch during training.

**Step 4**: Apply the Reparameterization Trick  

Because the sampling operation $z \sim q_\phi(z \mid x)$ is not differentiable with respect to the encoder parameters $\phi$, we cannot backpropagate through it directly. 
The Reparameterization Trick solves this by expressing $z$ as a differentiable transformation of a noise variable $\epsilon$ that is independent of $\phi$:
$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

Now that we have a random sample of $z_\phi$ from the encoder, we can feed it to the decoder to get $\mu_\theta(z)$, and evaluate:
$$
\mathbb{E}_{q_{\phi }(z\mid x)}[\log p_{\theta }(x\mid z)]\approx -\frac{D}{2}\log (2\pi \sigma ^{2})-\frac{1}{2\sigma ^{2}}\|{}x-\mu _{\theta }(z_\phi)\|{}^{2}
$$

- Note: We use the notation $z_\phi$ to indicate that the sample $z$ is drawn from the encoder distribution parameterized by $\phi$.

**Step 5**: Drop Constants to get the VAE Loss

We can drop the terms that do not contain learnable weights ($\theta $ or $\phi $):
1. $-\frac{D}{2}\log(2\pi\sigma^2)$ is a constant. Its derivative with respect to network parameters is 0. We drop it.
2. The fixed variance is chosen to be $\sigma^2 = 0.5$. This causes the scaling factor $\frac{1}{2\sigma ^{2}}$ to become exactly $\frac{1}{2(0.5)} = 1$.

Thus, 
$$
\mathbb{E}_{q_{\phi }(z\mid x)}[\log p_{\theta }(x\mid z)]\approx -\|{}x-\mu _{\theta }(z_\phi)\|{}^{2}
$$

3. Since we want to minimize the negative ELBO, the reconstruction loss is defined as the squared error between the input and the reconstructed output:

$$
\text{Reconstruction\ Loss} = \|{}x-\mu _{\theta }(z_\phi)\|{}^{2}
$$
This is precisely the Sum of Squared Errors (SSE) or Mean Squared Error (MSE) between the input data and decoded data.
