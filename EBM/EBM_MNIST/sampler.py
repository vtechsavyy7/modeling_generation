from torch import device, nn
import torch

import random
import numpy as np

class LangevinSampler(nn.Module):
    """Generate images from an energy-based model using Langevin dynamics.

    Previously generated samples are maintained in a buffer and reused to
    initialize new sampling runs.
    """

    def __init__(self, energy_function, img_shape, batch_size, max_len=8192):
        """Initialize the sampler and its sample buffer.

        Args:
            energy_function: Neural network that computes an energy for each
                input image.
            img_shape: Shape of an individual image, excluding the batch dimension.
            batch_size: Number of images to generate in each batch.
            max_len: Maximum number of samples to retain in the buffer.
        """
        super().__init__()
        self.energy_function = energy_function
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.max_len = max_len

        # Store the device of the energy function for later use when generating samples (to ensure that the samples are generated on the same device as the model).
        self.device = device("cpu")  # Default to CPU if energy_function has no parameters
        if hasattr(energy_function, 'parameters') and len(list(energy_function.parameters())) > 0:
            self.device = next(energy_function.parameters()).device

        # Buffer for storing previously generated samples. Initialized with random noise with values scaled to [-1, 1].
        # TODO: Maybe replace the list with a more efficient data structure, e.g. a queue or a ring buffer.
        self.buffer = [(torch.rand((1,)+img_shape, device=self.device)*2-1) for _ in range(self.batch_size)]

    def sample_new_exmps(self, steps=60, step_size=10):
        """Generate a batch of images and add them to the sample buffer.

        Args:
            steps: Number of Langevin dynamics iterations.
            step_size: Scale of each gradient update.

        Returns:
            A tensor of generated images with shape ``(batch_size, *img_shape)``.
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.batch_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape, device=self.device)
        rand_imgs = rand_imgs * 2 - 1 # Scale to [-1, 1]
        old_imgs = torch.cat(random.choices(self.buffer, k=self.batch_size-n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(self.device)

        # Perform MCMC sampling
        inp_imgs = LangevinSampler.generate_samples(self.energy_function, inp_imgs, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.buffer = list(inp_imgs.to(self.device).chunk(self.batch_size, dim=0)) + self.buffer
        self.buffer = self.buffer[:self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(energy_function, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
        """Sample images using Langevin dynamics for the given energy function.

        Args:
            energy_function: Neural network that computes an energy for each
                input image.
            inp_imgs: Initial image tensor, updated in place during sampling.
            steps: Number of Langevin dynamics iterations.
            step_size: Scale of each gradient update.
            return_img_per_step: Whether to return images from every iteration
                instead of only the final images.

        Returns:
            The final image tensor, or a tensor of images stacked by iteration
            with shape ``(steps, *inp_imgs.shape)`` if return_img_per_step is
            true.
        """
        # Store the training state of the model to restore it later
        is_training = energy_function.training

        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        energy_function.eval()
        for p in energy_function.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            sample_energies = energy_function(inp_imgs)  # [batch_size, 1] (some tutorials add a minus sign here, and then flip the signs in the loss function)
            sample_energies.mean().backward() # We take the mean over the batch to get a single scalar value to backpropagate from. This is equivalent to summing the gradients over the batch, which is what we want for MCMC sampling.
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data) # Negative sign because we want to move in the direction of lower energy

            # Detach gradients and zero them for the next iteration, also clamp the pixel values to [-1, 1]
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in energy_function.parameters():
            p.requires_grad = True
        energy_function.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs