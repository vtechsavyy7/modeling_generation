import torch 


class DiffusionForward:
    def __init__(self, schedule_type= "linear", num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """Class for implementing the forward diffusion process in a diffusion model.
        
        Args:
            schedule_type (str): The type of noise schedule to use. Options are "linear" and "cosine".
            num_timesteps (int): The number of timesteps in the diffusion process.
            beta_start (float): The starting value of beta for the noise schedule.
            beta_end (float): The ending value of beta for the noise schedule.
        """
        self.schedule_type = schedule_type
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Generate the noise schedule (betas) based on the specified schedule type and parameters
        if self.schedule_type == "linear":
            self.betas = self.linear_schedule_betas()
        elif self.schedule_type == "cosine":
            self.betas = self.cosine_schedule_betas()

        # Precompute the cumulative products of alphas for efficient sampling during the forward diffusion process
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        # Pre-compute some quatities that will be useful for the reverse diffusion process (sampling) to avoid redundant computations during training and sampling.
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas = torch.sqrt(1.0 / self.alphas - 1)

    
    def linear_schedule_betas(self):
        """Generate a linear noise schedule for the diffusion process.
        
        Returns:
            torch.Tensor: A tensor of shape (num_timesteps,) containing the noise levels for each timestep.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
    
    def cosine_schedule_betas(self):
        """Generate a cosine noise schedule for the diffusion process.
        
        Returns:
            torch.Tensor: A tensor of shape (num_timesteps,) containing the noise levels for each timestep.
        """
        timesteps = torch.arange(self.num_timesteps + 1, dtype=torch.float64) / self.num_timesteps
        alphas_cumprod = torch.cos((timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to start at 1
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.999)
    

    def generate_noised_data(self, x0, t, noise):
        """Apply the forward diffusion process to the input data at a given timestep.
        
        Args:
            x0 (torch.Tensor): The original data tensor of shape (batch_size, channels, height, width).
            t (torch.Tensor): A tensor of shape (batch_size,) containing the timesteps for each sample in the batch.
            noise (torch.Tensor): A tensor of shape (batch_size, channels, height, width) containing the noise to be added to the data.
        
        Returns:
            torch.Tensor: The noised data tensor of shape (batch_size, channels, height, width) after applying the forward diffusion process.
        """
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)  # Get sqrt(alpha_bar) for timestep t
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)  # Get sqrt(1 - alpha_bar) for timestep t
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise  # Combine original data with noise according to the schedule
    
        

    