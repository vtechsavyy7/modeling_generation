## Score Matching Loss

The score matching loss is used to train models to match the score function of a data distribution.

### Resources: 

- [Hyvärinen, A. (2005). Estimation of non-normalized statistical models by score matching.](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)
- [Vincent, P. (2011). A connection between score matching and denoising autoencoders.](https://proceedings.mlr.press/v15/vincent11a.html)

### Definitions: 

Let $p(x)$ be the data distribution. The true score function is defined as the gradient of the log-probability density with respect to the input:

$$
s(x) = \nabla_x \log p(x)
$$

Let $s_\phi(x)$ be the model's estimate of the score function:

$$
s_\phi(x) \approx s(x)
$$

### Score Matching Loss:

The score matching loss is defined as the expected squared difference between the true score function and the model's estimate:

$$
\mathcal{L}_{SM}(\phi) = \mathbb{E}_{x \sim p(x)} \left[ \| s_\phi(x) - s(x) \|_2^2 \right]
$$

The true score function $s(x)$ is generally unknown, so the score matching loss is typically rewritten in a form that depends only on the model's estimate $s_\phi(x)$ and its derivatives, using integration by parts and properties of the score function.

### Tractable Form of the Score Matching Loss

