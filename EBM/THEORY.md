# Energy Based Modeling

## Questions: 
- How do EBMs handle the normalization of probability distributions?
  - For optimizing the parameters of an EBM, we need to compute the gradient of the log-likelihood with respect to the model parameters. 
  - This involves the normalization constant, which is often intractable to compute directly.
  - **Contrastive Divergence** is the technique commonly used to approximate the gradient of the log-likelihood, which helps in handling the normalization of probability distributions in EBMs.