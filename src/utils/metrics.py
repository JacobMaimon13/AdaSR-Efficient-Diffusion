import torch
import numpy as np

try:
    import piq
    import lpips
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: 'piq' or 'lpips' not found. Metrics will be disabled.")

class MetricCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_model = None
        if METRICS_AVAILABLE:
            try:
                self.lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
            except:
                print("Could not initialize LPIPS model.")

    def calculate(self, sr, hr):
        """
        Calculate PSNR, SSIM, and LPIPS for a batch of images.
        Expects input in range [0, 1].
        """
        if not METRICS_AVAILABLE:
            return {}

        # Move to device
        sr = sr.to(self.device)
        hr = hr.to(self.device)

        metrics = {}
        
        # PSNR
        metrics['psnr'] = piq.psnr(sr, hr, data_range=1.0, reduction='mean').item()
        
        # SSIM
        metrics['ssim'] = piq.ssim(sr, hr, data_range=1.0, downsample=False, reduction='mean').item()
        
        # LPIPS (Expects input in [-1, 1])
        if self.lpips_model is not None:
            sr_norm = sr * 2.0 - 1.0
            hr_norm = hr * 2.0 - 1.0
            metrics['lpips'] = self.lpips_model(sr_norm, hr_norm, normalize=True).mean().item()
            
        return metrics
