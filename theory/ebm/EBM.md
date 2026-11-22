# Energy Based Modeling

## Problem Formulation

In Energy Based Models (EBMs), we define an energy function $E_\phi(x)$ parameterized by $\phi$.  </br>
The energy function assigns a scalar energy to each configuration, where lower energy corresponds to more probable configurations. </br>
The probability distribution over the data is defined as:

$$
p_\phi(x) = \frac{\exp(-E_\phi(x))}{Z_\phi}
$$

with 

$$
Z_\phi = \int \exp(-E_\phi(x)) dx
$$

Here, $Z_\phi$ is the normalization constant (partition function), which is generally intractable to compute directly. </br>
The goal is to learn the parameters $\phi$ such that the model distribution $p_\phi(x)$ approximates the true data distribution $p_{\text{data}}(x)$.


## Learning the Parameters of the Model

The parameters $\phi$ are typically learned by maximizing the log-likelihood of the observed data:

$$
\mathcal{L}(\phi) = \sum_{i=1}^N \log p_\phi(x_i) = \sum_{i=1}^N \left[-E_\phi(x_i) - \log Z_\phi \right]
$$

Let's look at the log-likelihood for a single data point:

$$
\log p_\phi(x) = -E_\phi(x) - \log Z_\phi
$$

Since minimizing a loss is easier, let's define the negative log-likelihood as the loss function:

$$
\mathcal{L}_{\text{NLL}}(\phi) = -\log p_\phi(x) = E_\phi(x) + \log Z_\phi
$$

The gradient of the negative log-likelihood with respect to $\phi$ for a single data point is:

$$
\nabla_\phi \mathcal{L}_{\text{NLL}}(\phi) = \nabla_\phi E_\phi(x) + \nabla_\phi \log Z_\phi
$$

The second term is tough to compute directly because it involves the gradient of the log of the partition function </br>
which contains an intractable integral over all possible configurations of $x$. </br>

In the next section, we will derive the gradient of the log partition function and show how it leads to the expectation term over the model distribution.

## Gradient of the Log Partition Function

Using the formula for the gradient of the logarithm, the gradient of the log partition function with respect to $\phi$ is:

$$
\nabla_\phi \log Z_\phi = \frac{1}{Z_\phi} \nabla_\phi Z_\phi
$$

Substituting the definition of $Z_\phi$, and taking the gradient operator inside the integral we get:
Note that we are assuming the interchange of the gradient and the integral is valid, which holds under certain regularity conditions.

$$
\nabla_\phi Z_\phi = \nabla_\phi \int \exp[-E_\phi(x)] dx = \int \nabla_\phi \exp[-E_\phi(x)] dx
$$

Using the chain rule of gradients for the exponential function, we have:

$$
\nabla_\phi \exp[-E_\phi(x)] = -\exp[-E_\phi(x)] \cdot \nabla_\phi E_\phi(x)
$$
Thus,

$$
\nabla_\phi Z_\phi = -\int \exp[-E_\phi(x)] \cdot \nabla_\phi E_\phi(x) dx
$$
and

$$
\nabla_\phi \log Z_\phi = -\frac{1}{Z_\phi} \int \exp[-E_\phi(x)] \cdot \nabla_\phi E_\phi(x) dx
$$

The partition function $Z_\phi$ can be taken inside the integral since it is a constant with respect to $x$, allowing us to rewrite the expression as an expectation over the model distribution:

$$
\nabla_\phi \log Z_\phi = -\int \frac{\exp[-E_\phi(x)]}{Z_\phi} \cdot \nabla_\phi E_\phi(x) dx = -\mathbb{E}_{x \sim p_\phi(x)}[\nabla_\phi E_\phi(x)]
$$

Now, substituting this result back into the gradient of the negative log-likelihood, we get:

$$
\nabla_\phi \mathcal{L}_{\text{NLL}}(\phi) = \nabla_\phi E_\phi(x) - \mathbb{E}_{x \sim p_\phi(x)}[\nabla_\phi E_\phi(x)]
$$

This shows that the gradient consists of a **positive phase** term, which depends on the observed data, and a **negative phase** term, which depends on the model distribution.


## Approximating the Negative Phase

The negative phase term involves an expectation over the model distribution which is generally intractable to compute exactly.  </br> Therefore, we need to approximate it using sampling methods that use the Markov Chain Monte Carlo (MCMC) technique to draw samples from the model distribution.

Given $M$ samples $\{x^{(m)}\}_{m=1}^M$ drawn from the model distribution $p_\phi(x)$ using MCMC, we can approximate the negative phase expectation as:

$$
\mathbb{E}_{x \sim p_\phi(x)}[\nabla_\phi E_\phi(x)] \approx \frac{1}{M} \sum_{m=1}^M \nabla_\phi E_\phi(x^{(m)})
$$

### Generating Samples from the Model using MCMC

To generate samples from the model distribution $p_\phi(x)$, we can use MCMC methods such as Gibbs sampling or Langevin dynamics. The key idea is to construct a Markov chain whose stationary distribution is the target distribution $p_\phi(x)$. By running the chain for a sufficient number of steps, we can obtain samples that approximate the model distribution.

For example, in Langevin dynamics, the update rule for the samples is given by:

$$
x_{k+1} = x_k - \frac{\epsilon}{2} \nabla_x E_\phi(x_k) + \eta_k
$$
where $\epsilon$ is the step size, $k$ is the iteration index, and $\eta_k \sim \mathcal{N}(0, \epsilon)$ is Gaussian noise added at each step. 

After running this process for $K$ steps, the resulting $x_K$ can be used as a sample from the model distribution.

Notice that the gradient $\nabla_x E_\phi(x_k)$ is with respect to the sample $x_k$ and not the model parameters $\phi$. </br> This gradient guides the sample towards regions of lower energy, while the added noise ensures exploration of the energy landscape, allowing the Markov chain to approximate the target distribution effectively.

### Initialization of the Markov Chain
(TODO: Correct notation: Use superscript for MCMC samples)

The initialization of the Markov chain is an important consideration in MCMC sampling. A common approach is to start the chain from a random sample, often drawn from a simple distribution such as a Gaussian or uniform distribution. 

Running MCMC from pure random noise until it reaches true equilibrium takes thousands of steps, which is too slow for every training iteration. Researchers use three main tricks to initialize $x_0$ so $K$ can stay small (e.g., \(K = 20\) to \(60\) steps):

-  Standard Contrastive Divergence (CD-$K$): Initialize $x_0$ directly at a real training data point $x_0$. This is fast and keeps samples close to the data manifold, though it leaves unexplored "false minima" far away from real data unpenalized.

-  Persistent Contrastive Divergence (PCD): Maintain a persistent chain across training updates. Instead of resetting $x_0$ to real data at every step, initialize $x_0$ using the final sample $x_K$ from the previous training iteration. Because $\phi $ changes slowly, the chain stays near equilibrium over time.

- Replay Buffer (Modern SGLD): Store a large buffer (e.g., 10,000 samples) of past generated images. At each training step, initialize $x_0$ from the buffer 95% of the time and from pure uniform noise 5% of the time. This combines the speed of persistent chains with global exploration so the model suppresses false low-energy regions everywhere.

## Putting together in a training loop

During a single training step on a mini-batch of real data $ \{x_1, \dots, x_M\}$:

1. **Sample Negatives**: For each sample in the batch, initialize $\tilde{x}_0$ (via CD, PCD, or Replay Buffer) and run $K$ steps of MCMC (with $\phi$ frozen) to get $x_K$.
2. **Compute Surrogate Loss**: Pass both real and generated batches through the energy network to compute a scalar surrogate loss:

$$
\mathcal{L}_{\text{surrogate}}(\phi)=\frac{1}{M}\sum _{i=1}^{M}E_{\phi }(x_{i})-\frac{1}{M}\sum _{i=1}^{M}E_{\phi }(x_{i}^{(K)})
$$
3. **Backpropagate**: Call automatic differentiation on $\mathcal{L}_{\text{surrogate}}(\phi)$ with respect to $\phi$ (treating $x_{i}^{(K)}$ as fixed constants, i.e., detach() in PyTorch) and update $\phi$ via Adam or SGD.


