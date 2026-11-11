
## Problem Formulation

- The objective of generative modeling is to approximate a probability distrbution from a large number of observed data samples 
  that are coming from some underlying true probability distribution.
- Let's denote the approximated probability distribution by $p_\theta(x)$, where $\theta$ represents the parameters of the model. 
- The goal is to make $p_\theta(x)$ as close as possible to the true underlying distribution $p_{\text{data}}(x)$.

- In Generative Modeling, we often use a neural network to parameterize $p_\theta(x)$ and optimize the parameters $\theta$ to minimize some divergence between $p_\theta(x)$ and $p_{\text{data}}(x)$.

$$
p_\theta(x) \approx p_{\text{data}}(x)
$$

## How to Measure Divergence

- How do we measure the difference between the approximated distribution $p_\theta(x)$ and the true distribution $p_{\text{data}}(x)$?
- The cross-entropy is a common measure used to quantify the difference between two probability distributions. For distributions $p_{\text{data}}(x)$ and $p_\theta(x)$, the cross-entropy is defined as:

$$
H(p_{\text{data}}, p_\theta) = -\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]
$$

- The expectation integral in the cross-entropy definition is taken over the true data distribution $p_{\text{data}}(x)$. In practice, since we have a finite set of observed data samples $\{x_1, x_2, ..., x_n\}$, we can approximate the expectation using the empirical average:

$$
H(p_{\text{data}}, p_\theta) \approx -\frac{1}{n} \sum_{i=1}^{n} \log p_\theta(x_i)
$$

- But now the next question is: How do we define the distribution $p_\theta(x)$ using a neural network such that it satisfies the properties of a valid probability distribution?
- One common approach is to use a neural network to output the parameters of a known probability distribution (e.g., mean and variance for a Gaussian distribution) and then define $p_\theta(x)$ based on these parameters. But if we do this we are constraining the form of the approximate distribution to a specific family, which may not fully capture the complexity of the true data distribution.
- A Gaussian distribution is by definition unimodal, meaning it has a single peak. This may not be sufficient to capture the complexity of real-world data distributions, which can be multimodal and have intricate structures.
- Question: What if the neural network instead outputted values that are proportional the probability densities of samples, then coudl we maximize these values for the observed data to approximate the true distribution? This idea leads to the concept of Maximum Likelihood Estimation (MLE). The problem is that if the likelihoods are not normalized the neural network will cheat and assign arbitrarily large values to maximize the objective, without producing a valid probability distribution.


## How to get a neural network to approximate a valid probability distribution

- Non-negative: The probability assigned by the approximate distribution should be non-negative for all possible outcomes.
- Normalized: The probabilities should sum to 1 over all possible outcomes, ensuring a valid probability distribution.
- Smoothness: The approximate distribution should not have abrupt changes for similar outcomes, promoting generalization and stability.






