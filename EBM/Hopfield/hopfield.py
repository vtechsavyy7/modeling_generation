import torch
from torch.nn import functional as F

# Create a Hopfield network class that inherits from torch.nn.Module
class Hopfield(torch.nn.Module):
    def __init__(self, input_size, threshold=0.0, max_iter=100):
        super(Hopfield, self).__init__()
        self.input_size = input_size
        self.threshold = threshold
        self.max_iter = max_iter
        # Initialize the weight matrix with zeros
        # Hopfield is a fully connected network, so the weight matrix is of size (input_size, input_size)
        self.weight_matrix = torch.zeros(input_size, input_size)

    def _energy(self, x):
        """Calculate the energy of the current state x.

        Energy is the sum of the products of the weights and the states. For a Hopfield network, the energy can be calculated as:
        E = -0.5 * sum(w_ij * x_i * x_j) for all i, j
        where w_ij is the weight between neuron i and j, and x_i, x_j are the states of neuron i and j respectively. 
        Args:
            x (torch.Tensor): Current state of the network."""
        energy = -0.5 * x @ self.weight_matrix @ x
        return energy
        
    def _sync_run(self, x, eps=1e-6):
        """Synchronously update

        Args:
            x (torch.Tensor): inputs
            eps (float): For checking convergence to local energy minimum. Defaults to 1e-6.

        """
        e = self._energy(x)

        for _ in range(self.max_iter):

            x = torch.sign(
                F.linear(x, self.weight_matrix) 
                    - self.threshold)

            new_e = self._energy(x)

            # Check for convergence to a local energy minimum
            if abs(new_e - e) < eps:
                print(f"Converged to local energy minimum after {_} iterations.")
                return x

            e = new_e
        return x

    def forward(self, x):
        assert x.ndim == 1
        
        return self._sync_run(x)