"""
Core DDPM math: the beta/alpha schedule, the forward noising process q(x_t | x_0),
recovering an x0 estimate from a noise prediction, and the reverse (sampling) loop.

Kept separate from the model and training loop so both train.py and any inference
script can import GaussianDiffusion without duplicating this logic.
"""

import torch


class GaussianDiffusion:
    def __init__(self, timesteps=500, beta_start=1e-4, beta_end=0.02, schedule="linear", device="cpu"):
        self.timesteps = timesteps
        self.device = device

        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        else:
            raise ValueError(f"Unsupported beta schedule: {schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute commonly used terms
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        # For the reverse process: posterior variance beta_tilde_t
        alpha_bars_prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])
        self.posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)

    def _extract(self, arr, t, x_shape):
        """Gather values from a 1D schedule tensor `arr` at indices `t`, reshaped to broadcast against x_shape."""
        out = arr.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise):
        """Forward process: sample x_t ~ q(x_t | x0) given noise ~ N(0, I)."""
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def predict_x0_from_noise(self, x_t, t, predicted_noise):
        """Invert q_sample to recover an estimate of x0 given the model's predicted noise."""
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bars, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        return x0_hat.clamp(-1.0, 1.0)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, class_ids, device):
        """
        Full reverse diffusion sampling loop: start from pure noise x_T and iteratively
        denoise down to x_0, using the model's noise prediction at each step.

        Args:
            model: the ConditionalDDPM network.
            shape: (B, C, H, W) shape of the samples to generate.
            class_ids: (B,) tensor of class labels to condition generation on.
            device: torch device.
        Returns:
            x0: (B, C, H, W) generated samples in [-1, 1].
        """
        model.eval()
        x_t = torch.randn(shape, device=device)
        B = shape[0]

        for t_step in reversed(range(self.timesteps)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            predicted_noise = model(x_t, class_ids, t.float())

            alpha_t = self._extract(self.alphas, t, x_t.shape)
            alpha_bar_t = self._extract(self.alpha_bars, t, x_t.shape)
            beta_t = self._extract(self.betas, t, x_t.shape)

            # Predicted mean of p(x_{t-1} | x_t)
            model_mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
            )

            if t_step > 0:
                posterior_var_t = self._extract(self.posterior_variance, t, x_t.shape)
                noise = torch.randn_like(x_t)
                x_t = model_mean + torch.sqrt(posterior_var_t) * noise
            else:
                x_t = model_mean  # no noise added on the final step

        model.train()
        return x_t.clamp(-1.0, 1.0)
