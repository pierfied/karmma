import numpy as np
import torch

from .HMCSampler import HMCSampler


class KarmmaSampler:
    def tensorize(self, x):
        return torch.Tensor(x).to(dtype=torch.float32, device=self.device)

    def __init__(self, g1_obs, g2_obs, mask, u, s, mu, shift, k2g1, k2g2, noise_cov_1,
                 noise_cov_2, device='cuda'):
        self.device = device

        self.mask = mask

        self.g1_obs = self.tensorize(g1_obs[mask])
        self.g2_obs = self.tensorize(g2_obs[mask])
        self.u = self.tensorize(u)
        self.s = self.tensorize(s)
        self.mu = self.tensorize(mu)
        self.shift = shift
        self.k2g1 = self.tensorize(k2g1)
        self.k2g2 = self.tensorize(k2g2)
        self.inv_noise_1 = self.tensorize(np.linalg.inv(noise_cov_1))
        self.inv_noise_2 = self.tensorize(np.linalg.inv(noise_cov_2))

    def prior(self, x):
        return -0.5 * torch.sum(torch.square(x) / self.s)

    def likelihood(self, g1, g2):
        like1 = -0.5 * (g1 - self.g1_obs) @ self.inv_noise_1 @ (g1 - self.g1_obs)
        like2 = -0.5 * (g2 - self.g2_obs) @ self.inv_noise_2 @ (g2 - self.g2_obs)

        return like1 + like2

    def posterior(self, x):
        k = torch.exp(self.mu + self.u @ x) - self.shift

        g1 = self.k2g1 @ k
        g2 = self.k2g2 @ k

        return self.prior(x) + self.likelihood(g1, g2)

    def transform(self, x):
        k = torch.exp(self.mu + self.u @ x) - self.shift

        return k

    def sample(self, num_burn, num_burn_steps, burn_epsilon, num_samples, num_samp_steps, samp_epsilon):
        x0 = torch.randn(len(self.s), device=self.device) * torch.sqrt(self.s)

        hmc = HMCSampler(self.posterior, x0, 1 / self.s, transform=self.transform, device=self.device)

        hmc.sample(num_burn, num_burn_steps, burn_epsilon)

        chain = hmc.sample(num_samples, num_samp_steps, samp_epsilon)

        return chain
