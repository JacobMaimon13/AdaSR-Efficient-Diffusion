from .complexity import RegionComplexityEstimator
from .allocation import AdaptiveStepAllocator
from .early_exit import AdaptiveEarlyStopping

class AdaptiveDecoder:
    """
    Main adaptive decoding coordinator
    Combines complexity estimation, step allocation, and early stopping
    """

    def __init__(
        self,
        complexity_method='gradient',
        min_steps=50,
        max_steps=300,
        patch_size=32,
        early_stopping_enabled=True,
        early_stopping_threshold=0.05,
        **kwargs
    ):
        # Filter kwargs for allocator (prevent passing early stopping params to allocator)
        allowed_allocator_params = ['complexity_threshold_low', 'complexity_threshold_high', 'allocation_method']
        allocator_kwargs = {k: v for k, v in kwargs.items() if k in allowed_allocator_params}

        self.complexity_estimator = RegionComplexityEstimator(
            method=complexity_method,
            patch_size=patch_size
        )
        self.step_allocator = AdaptiveStepAllocator(
            min_steps=min_steps,
            max_steps=max_steps,
            patch_size=patch_size,
            **allocator_kwargs
        )
        self.early_stopping = AdaptiveEarlyStopping(
            enabled=early_stopping_enabled,
            quality_threshold=early_stopping_threshold
        )
        self.patch_size = patch_size

    def estimate_complexity(self, image, **kwargs):
        """Estimate complexity map for the image"""
        return self.complexity_estimator(image, **kwargs)

    def allocate_steps(self, complexity_map):
        """Allocate steps based on complexity"""
        return self.step_allocator.allocate_steps(complexity_map)

    def should_stop_early(self, current_step, current_image, reference_image=None):
        """Check if we should stop early"""
        return self.early_stopping.should_stop(current_step, current_image, reference_image)
