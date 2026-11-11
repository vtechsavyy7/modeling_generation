# Score Function


## Overview

The score function is a fundamental concept in score-based generative models and energy-based models. It is defined as the gradient of the log-probability density function with respect to the input data:

For a given probability density function $p(x)$ on $\mathbb{R}^D$, the score function is defined as:

$$
s(x) = \nabla_x \log p(x),    s: \mathbb{R}^D \to \mathbb{R}^D
$$