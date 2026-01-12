import torch

class AdaptiveEarlyStopping:
    """
    Implements early stopping based on quality thresholds
    """

    def __init__(
        self,
        enabled=True,
        quality_metric='lpips',
        quality_threshold=0.05,
        check_interval=10,
        min_steps=50
    ):
        self.enabled = enabled
        self.quality_metric = quality_metric
        self.quality_threshold = quality_threshold
        self.check_interval = check_interval
        self.min_steps = min_steps

        if quality_metric == 'lpips':
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex', verbose=False)
            except ImportError:
                print("Warning: lpips not available, using pixel-based metric")
                self.quality_metric = 'pixel'
                self.lpips_model = None
        else:
            self.lpips_model = None

    def compute_quality(self, current_image, reference_image=None):
        """
        Compute quality metric for current image
        If reference is None, uses self-consistency metrics
        """
        if self.quality_metric == 'lpips' and self.lpips_model is not None and reference_image is not None:
            # Compare with reference (e.g., LR upscaled image)
            with torch.no_grad():
                # Ensure lpips_model is on the same device as the images
                device = current_image.device
                if next(self.lpips_model.parameters()).device != device:
                    self.lpips_model = self.lpips_model.to(device)

                # Normalize to [-1, 1] for LPIPS
                ref_norm = (reference_image * 2.0) - 1.0
                curr_norm = (current_image * 2.0) - 1.0
                quality = self.lpips_model(ref_norm, curr_norm).mean()
                return 1.0 - quality.item()  # Convert distance to similarity
        elif self.quality_metric == 'pixel':
            # Use pixel variance as proxy for quality (smooth images = low variance)
            variance = current_image.var(dim=1).mean()
            return variance.item()
        else:
            # Default: use gradient magnitude as quality proxy
            grad = torch.abs(current_image[:, :, 1:, :] - current_image[:, :, :-1, :])
            quality = grad.mean().item()
            return quality

    def should_stop(self, current_step, current_image, reference_image=None):
        """
        Determine if we should stop early based on quality
        """
        if not self.enabled:
            return False

        if current_step < self.min_steps:
            return False

        # Check quality at intervals
        if current_step % self.check_interval != 0:
            return False

        # Ensure images are on same device
        if reference_image is not None and reference_image.device != current_image.device:
            reference_image = reference_image.to(current_image.device)

        quality = self.compute_quality(current_image, reference_image)

        # Stop if quality is above threshold (good enough)
        return quality >= self.quality_threshold
