
## 2026-11-07: 

- Problem Formulation
  - DDPM paper formulates the problem as a parameterized Markov chain that gradually denoises a sample from a Gaussian distribution to generate data.
  - The transitions along this Markov chain are what the neural network has to learn.

- TODO: 
  - Write down the mathematical formulation of the Markov chain transitions.


## 2026-11-11: 

- Trying to understand Diffusion Models from the Score-Based perspective.
- Starting with Energy-Based Models (EBMs): 
  - In EBMs, Constrative Divergence is used to approximate the loss term that involves the intractable partition function.
  - What does Contrastive Divergence actually compute and why is it used in EBMs?
