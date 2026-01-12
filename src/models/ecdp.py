import torch
import torch.nn as nn
import torchvision.transforms.functional as FV
import math

from .unet import UNet
from .rrdb import RRDBEncoder
from .diffusion import Diffusion
from ..adaptive.decoder import AdaptiveDecoder # Import from your adaptive module

class ImageSizeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size_x = None
        self.image_size_y = None

    def set_working_image_size(self, size):
        self.image_size_x, self.image_size_y = size

class ImageSizeManagerMixin(ImageSizeMixin):
    def set_image_size(self, sizes):
        def work(module):
            if isinstance(module, ImageSizeMixin):
                module.set_working_image_size(sizes)
        self.apply(work)

class ECDP(ImageSizeMixin, nn.Module):
    def __init__(self, input_channels=3, sr_factor=4, rrdb_num_blocks=23,
                 rrdb_block_selection=[1, 8, 15, 22], rrdb_network_features=64,
                 rrdb_intermediate_features=32, rrdb_channels=256, use_pretrained_rrdb=False,
                 use_adaptive_decoding=False, adaptive_decoding_config=None):
        super().__init__()
        self.in_channels = input_channels
        self.diffusion = Diffusion(UNet(self.in_channels, 128, rrdb_channels, out_channels=2 * self.in_channels))
        self.lr_feats = RRDBEncoder(self.in_channels, rrdb_num_blocks, rrdb_block_selection,
                                    rrdb_network_features, rrdb_intermediate_features)
        
        # Adaptive Decoding Setup
        self.use_adaptive_decoding = use_adaptive_decoding
        self.adaptive_decoder = None
        if self.use_adaptive_decoding:
            if adaptive_decoding_config is None: adaptive_decoding_config = {}
            self.adaptive_decoder = AdaptiveDecoder(**adaptive_decoding_config)
            self.diffusion.set_adaptive_decoding(self.adaptive_decoder)

    def forward(self, *args, mode, **kwargs):
        if mode == "loss":
            return self._calculate_loss(*args, **kwargs)
        elif mode == "generate":
            return self._generate_sample(*args, **kwargs)
        else:
            raise ValueError("invalid forward mode")

    def _calculate_loss(self, x, *, cond):
        lr_feats = self.lr_feats(cond)
        cond_scaled = FV.resize(cond, [x.shape[2], x.shape[3]], interpolation=FV.InterpolationMode.BICUBIC)
        scale = 5
        x = x - cond_scaled
        x = x * scale
        return self.diffusion.normalize(x, cond=(lr_feats, cond_scaled, scale))

    def _generate_sample(self, x, *, cond):
        x = x.view(x.shape[0], self.in_channels, self.image_size_x, self.image_size_y)
        lr_feats = self.lr_feats(cond)
        cond_scaled = FV.resize(cond, [x.shape[2], x.shape[3]], interpolation=FV.InterpolationMode.BICUBIC)
        scale = 5
        
        # Pass cond_scaled as reference for adaptive stopping
        x = self.diffusion.generate(x, cond=(lr_feats, cond_scaled, scale), reference_image=cond_scaled)
        
        x = x / scale
        x = x + cond_scaled
        return x

    def set_generate_steps(self, steps):
        self.diffusion.set_generate_steps(steps)

class GaussianPrior(ImageSizeMixin, nn.Module):
    def __init__(self, shape_gen):
        super().__init__()
        self.shape_gen = shape_gen
    
    @property
    def latent_size(self):
        return self.shape_gen(self.image_size_x, self.image_size_y)

    def sample(self, batch_size, *, device=None):
        return torch.randn((batch_size, self.latent_size), device=device)

class ConditionalDensityModel(ImageSizeManagerMixin, nn.Module):
    def __init__(self, input_channels=3, sr_factor=4, **kwargs):
        super().__init__()
        self.model = ECDP(input_channels=input_channels, sr_factor=sr_factor, **kwargs)
        self.prior = GaussianPrior(lambda x, y: input_channels * x * y)

    def forward(self, *args, mode, **kwargs):
        return self.model(*args, mode=mode, **kwargs)
        
    def set_generate_steps(self, steps):
        self.model.set_generate_steps(steps)
