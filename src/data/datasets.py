import torch
import numpy as np
import torchvision.transforms.functional as FV
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

# --- Helper Functions ---

def imresize(image, output_shape):
    """Simple bicubic resize using torchvision"""
    if isinstance(image, np.ndarray):
        original_dtype = image.dtype
        # Convert to tensor
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().unsqueeze(0)
        
        if isinstance(output_shape, tuple):
            output_shape = list(output_shape)
            
        resized = FV.resize(image_tensor, output_shape, interpolation=FV.InterpolationMode.BICUBIC)
        resized_np = resized.squeeze(0).permute(1, 2, 0).numpy()
        
        if original_dtype == np.uint8:
            resized_np = np.clip(resized_np, 0, 255)
            return resized_np.astype(np.uint8)
        return resized_np
        
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif len(image.shape) == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        if isinstance(output_shape, tuple):
            output_shape = list(output_shape)
            
        resized = FV.resize(image.float(), output_shape, interpolation=FV.InterpolationMode.BICUBIC)
        return resized.squeeze(0) if resized.shape[0] == 1 else resized
    return image

def image_to_tensor(image):
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        image = image.transpose(2, 0, 1).copy()
        image = torch.from_numpy(image)
        image = image.to(torch.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif len(image.shape) == 3 and image.shape[0] != 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        if image.max() > 1.0:
            image = image / 255.0
    return image

def image_to_hr_lr_tensor(image, downscale_factor):
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 4: image = image[0]
        if len(image.shape) == 3: image = image.permute(1, 2, 0).numpy()
        else: image = image.numpy()
        
    h, w = image.shape[:2]
    image_lr = imresize(image, [h // downscale_factor, w // downscale_factor])
    image = image_to_tensor(image)
    image_lr = image_to_tensor(image_lr)
    return {"image": image, "image_lr": image_lr}

# --- Datasets ---

class DIV2KTestDataset(Dataset):
    """DIV2K validation set for 4x super-resolution testing"""
    def __init__(self, data_dir=None, crop_size=None):
        if data_dir is None: data_dir = Path("data")
        else: data_dir = Path(data_dir)

        self.hr_dir = data_dir / "div2k" / "DIV2K_valid_HR"
        self.lr_dir = data_dir / "div2k" / "DIV2K_valid_LR_bicubic" / "X4"
        self.data_range = range(801, 901)
        self.crop_size = crop_size

        if not self.hr_dir.exists() or not self.lr_dir.exists():
            raise FileNotFoundError(f"DIV2K dataset not found in {data_dir}")

    def __len__(self):
        return len(self.data_range)

    def __getitem__(self, idx):
        num = self.data_range[idx]
        hr_path = self.hr_dir / f"{num:04d}.png"
        lr_path = self.lr_dir / f"{num:04d}x4.png"

        with Image.open(hr_path) as f: image_hr = np.asarray(f.convert('RGB'))
        with Image.open(lr_path) as f: image_lr = np.asarray(f.convert('RGB'))

        if self.crop_size is not None:
            h, w = image_hr.shape[:2]
            start_h = (h - self.crop_size) // 2
            start_w = (w - self.crop_size) // 2
            image_hr = image_hr[start_h:start_h+self.crop_size, start_w:start_w+self.crop_size]
            
            lr_crop = self.crop_size // 4
            h_lr, w_lr = image_lr.shape[:2]
            start_h_lr = (h_lr - lr_crop) // 2
            start_w_lr = (w_lr - lr_crop) // 2
            image_lr = image_lr[start_h_lr:start_h_lr+lr_crop, start_w_lr:start_w_lr+lr_crop]

        return {"image": image_to_tensor(image_hr), "image_lr": image_to_tensor(image_lr)}

class ImageFolders(Dataset):
    """Base class for loading images from folders (for training)"""
    def __init__(self, paths, downscale_factor, random_crop_size=None, deterministic=False, repeat=1):
        self.files = []
        for path in paths:
            path = Path(path)
            if not path.exists(): continue
            self.files.extend([f for f in path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        
        self.files.sort()
        self.downscale_factor = downscale_factor
        self.random_crop_size = random_crop_size
        self.deterministic = deterministic
        self.repeat = repeat

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        path = self.files[idx % len(self.files)]
        with Image.open(path) as f: image = np.asarray(f.convert('RGB'))

        if self.random_crop_size is not None:
            h, w = image.shape[:2]
            if h < self.random_crop_size or w < self.random_crop_size:
                # Resize if too small
                image = imresize(image, [max(h, self.random_crop_size), max(w, self.random_crop_size)])
            
            h, w = image.shape[:2]
            x_idx = np.random.randint(0, h - self.random_crop_size + 1)
            y_idx = np.random.randint(0, w - self.random_crop_size + 1)
            image = image[x_idx:x_idx + self.random_crop_size, y_idx:y_idx + self.random_crop_size, :]

        return image_to_hr_lr_tensor(image, self.downscale_factor)

class DF2kDataset(ImageFolders):
    """DF2K dataset (DIV2K + Flickr2K) for training"""
    def __init__(self, data_dir, downscale_factor, train=True, random_crop_size=None):
        data_dir = Path(data_dir)
        if train:
            paths = [data_dir / "div2k" / "DIV2K_train_HR", data_dir / "flickr2k" / "Flickr2K_HR"]
        else:
            paths = [data_dir / "div2k" / "DIV2K_valid_HR"]
        
        super().__init__(paths, downscale_factor, random_crop_size=random_crop_size)

class AugmentedDataset(Dataset):
    """Wrapper for data augmentation"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image, image_lr = data["image"], data["image_lr"]

        # Random Flip
        if torch.rand(()) < 0.5:
            image = FV.hflip(image)
            image_lr = FV.hflip(image_lr)

        # Random Rotate
        k = torch.randint(0, 4, size=()).item()
        image = torch.rot90(image, k, dims=[-2, -1])
        image_lr = torch.rot90(image_lr, k, dims=[-2, -1])

        return {"image": image, "image_lr": image_lr}

class ToyDataset(Dataset):
    """Synthetic dataset for testing code flow without real data"""
    def __init__(self, num=1000, image_size=160):
        self.num = num
        self.image_size = image_size
        self.downscale_factor = 4

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # Create random colorful image
        hr = torch.rand(3, self.image_size, self.image_size)
        return image_to_hr_lr_tensor(hr, self.downscale_factor)
