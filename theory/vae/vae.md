## Problem Formulation

Let $\bold{X} = \{x_1, x_2, ..., x_n\}$ be a set of observed data points, where each $x_i$ is a high-dimensional vector.  </br>

The goal of Variational Autoencoders (VAEs) is to learn a probabilistic model that can generate new data points similar to the observed data.
i.e we want to model the underlying distribution of the data and be able to sample from it. 

Let us denote the probabilistic model as $p_\theta(\mathbb{X})$, where $\theta$ represents the parameters of the model.