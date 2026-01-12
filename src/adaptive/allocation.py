import torch

class AdaptiveStepAllocator:
    """
    Allocates different numbers of diffusion steps to different image regions
    based on their complexity
    """

    def __init__(
        self,
        min_steps=50,
        max_steps=300,
        patch_size=32,
        complexity_threshold_low=0.3,
        complexity_threshold_high=0.7,
        allocation_method='linear'
    ):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.patch_size = patch_size
        self.complexity_threshold_low = complexity_threshold_low
        self.complexity_threshold_high = complexity_threshold_high
        self.allocation_method = allocation_method

    def allocate_steps(self, complexity_map):
        """
        Allocate diffusion steps for each patch based on complexity

        Args:
            complexity_map: [B, 1, H, W] complexity values in [0, 1]

        Returns:
            step_map: [B, 1, H, W] number of steps for each pixel
            patch_steps: dict mapping patch coordinates to step counts
        """
        B, C, H, W = complexity_map.shape

        # Divide into patches
        num_patches_h = (H + self.patch_size - 1) // self.patch_size
        num_patches_w = (W + self.patch_size - 1) // self.patch_size

        step_map = torch.zeros_like(complexity_map)
        patch_steps = {}

        for b in range(B):
            for ph in range(num_patches_h):
                for pw in range(num_patches_w):
                    h_start = ph * self.patch_size
                    h_end = min((ph + 1) * self.patch_size, H)
                    w_start = pw * self.patch_size
                    w_end = min((pw + 1) * self.patch_size, W)

                    # Get average complexity for this patch
                    patch_complexity = complexity_map[b, 0, h_start:h_end, w_start:w_end].mean()

                    # Allocate steps based on complexity
                    if self.allocation_method == 'linear':
                        # Linear interpolation between min and max steps
                        steps = int(
                            self.min_steps +
                            (self.max_steps - self.min_steps) * patch_complexity.item()
                        )
                    elif self.allocation_method == 'threshold':
                        # Threshold-based allocation
                        if patch_complexity < self.complexity_threshold_low:
                            steps = self.min_steps
                        elif patch_complexity > self.complexity_threshold_high:
                            steps = self.max_steps
                        else:
                            # Linear interpolation in middle range
                            t = (patch_complexity - self.complexity_threshold_low) / (
                                self.complexity_threshold_high - self.complexity_threshold_low + 1e-8
                            )
                            steps = int(self.min_steps + (self.max_steps - self.min_steps) * t)
                    elif self.allocation_method == 'exponential':
                        # Exponential allocation (more aggressive)
                        steps = int(
                            self.min_steps +
                            (self.max_steps - self.min_steps) * (patch_complexity ** 2).item()
                        )
                    else:
                        raise ValueError(f"Unknown allocation method: {self.allocation_method}")

                    # Assign steps to this patch region
                    step_map[b, 0, h_start:h_end, w_start:w_end] = steps
                    patch_steps[(b, ph, pw)] = steps

        return step_map, patch_steps
