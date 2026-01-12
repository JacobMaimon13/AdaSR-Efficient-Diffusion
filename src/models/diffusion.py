import torch
import torch.nn as nn

def make_rand_like(x):
    return torch.randn_like(x)

class Diffusion(nn.Module):
    def __init__(self, network):
        super().__init__()
        t_max = 300
        self.b_min = 0.1
        self.b_max = 20
        ts = torch.arange(0, t_max + 1) / t_max
        alpha_bars_ext = self.t_to_aa(ts)
        alphas = alpha_bars_ext[1:] / alpha_bars_ext[:-1]
        betas = 1 - alphas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars_ext[1:])
        self.network = network
        self.t_count = 0
        self.gen_steps = 200
        self.gen_verbose = False
        self.ddim = False
        
        # Adaptive decoding integration
        self.adaptive_decoding = None
        self.use_adaptive_decoding = False
        self.actual_steps_used = []

    def t_to_aa(self, ts):
        return torch.exp(-ts * (ts * (self.b_max - self.b_min) / 2 + self.b_min))

    def normalize(self, x, *, cond):
        cond, diff_mean, diff_scale = cond
        if self.training:
            ts = torch.rand(x.shape[0], device=x.device) * (len(self.alpha_bars) - 0) + 0
            sampled_aa = self.t_to_aa(ts / len(self.alpha_bars))
            self.t_count = 0
        else:
            ts = (torch.arange(self.t_count, self.t_count + x.shape[0], device=x.device) * 83 % len(self.alpha_bars)) + 1
            sampled_aa = self.alpha_bars[ts - 1]
            self.t_count += x.shape[0]
        eps = make_rand_like(x)
        xt = x * sampled_aa.sqrt()[:, None, None, None] + eps * (1 - sampled_aa).sqrt()[:, None, None, None]
        network_pred_eps_x0 = self.network(xt, ts, cond)
        network_pred_eps, network_pred_x0 = network_pred_eps_x0.split(network_pred_eps_x0.shape[1] // 2, dim=1)
        loss1 = (network_pred_eps - eps).square().sum(dim=[1, 2, 3])
        loss2 = (network_pred_x0 - x).square().sum(dim=[1, 2, 3])
        return loss1 + loss2

    def set_generate_steps(self, steps):
        self.gen_steps = steps

    def set_adaptive_decoding(self, adaptive_decoder):
        self.adaptive_decoding = adaptive_decoder
        self.use_adaptive_decoding = adaptive_decoder is not None

    def generate(self, x, *, cond, reference_image=None):
        # Dispatch to adaptive generation if enabled
        if self.use_adaptive_decoding and self.adaptive_decoding is not None:
            return self._generate_adaptive(x, cond=cond, reference_image=reference_image)
        
        # Standard generation
        cond_tuple, diff_mean, diff_scale = cond
        end_step = 0 if self.ddim else 5
        
        for t in range(self.gen_steps - 1, end_step - 1, -1):
            if t == self.gen_steps - 1:
                x = self._next_step(x, cond_tuple, t, make_rand_like(x), end_step)
            else:
                x = torch.utils.checkpoint.checkpoint(self._next_step, x, cond_tuple, t, make_rand_like(x), end_step)
        return x

    def _next_step(self, x, cond, t, the_eps, end_step):
        network_pred_eps_x0 = self.network(x, torch.tensor([t + 1], device=x.device).expand(x.shape[0]), cond)
        network_pred_eps, network_pred_x0 = network_pred_eps_x0.split(network_pred_eps_x0.shape[1] // 2, dim=1)
        network_pred = network_pred_eps * self.alpha_bars[t].sqrt() - network_pred_x0 * (1 - self.alpha_bars[t]).sqrt()
        eps_pred = x * (1 - self.alpha_bars[t]).sqrt() + network_pred * self.alpha_bars[t].sqrt()
        
        if self.ddim:
            eps_coeff = (1 - self.alphas[t]) / ((1 - self.alpha_bars[t]).sqrt() + (self.alphas[t] - self.alpha_bars[t]).sqrt())
            x = (x - eps_coeff * eps_pred) / self.alphas[t].sqrt()
        else:
            if t == end_step:
                x = (x - (1 - self.alpha_bars[t]).sqrt() * eps_pred) / self.alpha_bars[t].sqrt()
            else:
                eps_coeff = (1 - self.alphas[t]) / (1 - self.alpha_bars[t]).sqrt()
                x = (x - eps_coeff * eps_pred) / self.alphas[t].sqrt()
                x = x + the_eps * (self.get_inject_noise_var(t) ** 0.5)
        return x

    def get_inject_noise_var(self, t):
        return self.betas[t] * (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t])

    def _generate_adaptive(self, x, *, cond, reference_image=None):
        """Generation with Adaptive Early Stopping"""
        cond_tuple, diff_mean, diff_scale = cond
        end_step = 0 if self.ddim else 5
        steps_used = 0
        
        for t in range(self.gen_steps - 1, end_step - 1, -1):
            steps_used += 1
            if t == self.gen_steps - 1:
                x = self._next_step(x, cond_tuple, t, make_rand_like(x), end_step)
            else:
                x = torch.utils.checkpoint.checkpoint(self._next_step, x, cond_tuple, t, make_rand_like(x), end_step)
            
            # Check early stopping
            if self.adaptive_decoding.early_stopping.enabled:
                # Denormalize for check
                current_img = (x / diff_scale) + diff_mean if diff_scale is not None else x
                current_img = torch.clamp(current_img, 0, 1)
                
                # Use reference if provided (usually cond_scaled)
                ref = reference_image if reference_image is not None else current_img 
                
                if self.adaptive_decoding.should_stop_early(self.gen_steps - 1 - t, current_img, ref):
                    break
        
        self.actual_steps_used.append(steps_used)
        return x
