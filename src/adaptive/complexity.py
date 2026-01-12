import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionComplexityEstimator(nn.Module):
    """
    Estimates the complexity/uncertainty of image regions
    to determine how many diffusion steps each region needs
    """

    def __init__(self, method='gradient', patch_size=32):
        super().__init__()
        self.method = method
        self.patch_size = patch_size

        if method == 'learned':
            # Simple CNN to predict region complexity
            self.complexity_net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            )

    def compute_gradient_complexity(self, image):
        """Compute complexity based on image gradients"""
        # Convert to grayscale for gradient computation
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image

        # Compute gradients
        grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])

        # Pad to original size
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')

        # Total gradient magnitude
        gradient_mag = (grad_x + grad_y) / 2.0

        return gradient_mag

    def compute_variance_complexity(self, image):
        """Compute complexity based on local variance"""
        # Convert to grayscale
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image

        # Compute local variance using convolution
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=image.device) / (kernel_size ** 2)

        mean = F.conv2d(gray, kernel, padding=kernel_size // 2)
        variance = F.conv2d((gray - mean) ** 2, kernel, padding=kernel_size // 2)

        return variance

    def compute_uncertainty_from_diffusion(self, x_t, network_pred_eps, network_pred_x0, alpha_bar):
        """
        Compute uncertainty based on diffusion model predictions
        Higher uncertainty = more disagreement between predictions
        """
        # Reconstruct x0 from both predictions
        x0_from_eps = (x_t - (1 - alpha_bar).sqrt() * network_pred_eps) / alpha_bar.sqrt()

        # Uncertainty is the disagreement between predictions
        uncertainty = torch.abs(x0_from_eps - network_pred_x0).mean(dim=1, keepdim=True)

        return uncertainty

    def forward(self, image, x_t=None, network_pred_eps=None, network_pred_x0=None, alpha_bar=None):
        """
        Compute complexity map for the image
        Returns: complexity map [B, 1, H, W] with values in [0, 1]
        """
        if self.method == 'gradient':
            complexity = self.compute_gradient_complexity(image)
        elif self.method == 'variance':
            complexity = self.compute_variance_complexity(image)
        elif self.method == 'diffusion_uncertainty':
            if x_t is None or network_pred_eps is None or network_pred_x0 is None:
                # Fallback to gradient if diffusion info not available
                complexity = self.compute_gradient_complexity(image)
            else:
                complexity = self.compute_uncertainty_from_diffusion(
                    x_t, network_pred_eps, network_pred_x0, alpha_bar
                )
        elif self.method == 'learned':
            complexity = self.complexity_net(image)
        else:
            raise ValueError(f"Unknown complexity method: {self.method}")

        # Normalize to [0, 1]
        complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-8)

        return complexity
