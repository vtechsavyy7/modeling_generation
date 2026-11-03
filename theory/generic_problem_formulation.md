## The Generative Modeling Problem

We are given a finite set of samples from some high-dimensional distribution, and our goal is to learn a model that can generate new samples from the same distribution.

Let us denote the samples as $\bold{X} = \{x_1, x_2, ..., x_n\}$, where each $x_i$ is a high-dimensional vector.
The true form of the underlying distribution is unknown, but we can assume that it can be represented by a probability density function $p_{data}(\bold{X})$.

Another important assumption that we make is that the samples are independent and identically distributed (i.i.d), which means that each sample is drawn from the same distribution and is independent of the other samples.

We formulate the generative model as a parameterized probability distribution $p_\theta(\bold{X})$, where $\theta$ represents the parameters of the model. 

The goal is to find the parameters $\theta$ that make $p_\theta(\bold{X})$ as close as possible to the true data distribution $p_{data}(\bold{X})$.

More formally, we want to minimize some measure of divergence ($\mathcal{D}$) between the two distributions

$$
\theta^* = \arg\min_\theta \mathcal{D}(p_{data}(\bold{X}) || p_\theta(\bold{X}))   \tag{1}
$$