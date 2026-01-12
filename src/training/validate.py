import torch
import torchvision.transforms.functional as FV
from tqdm import tqdm
import time

def test_model(model, test_loader, device='cuda', save_images=False, output_dir='results'):
    model = model.to(device)
    model.eval()
    
    # Set default steps for testing
    if hasattr(model, 'set_generate_steps'):
        model.set_generate_steps(200)

    results = []
    
    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            lr = batch['image_lr'].to(device)
            hr_gt = batch['image'].to(device)

            # Set dynamic image size based on HR target
            _, _, h_hr, w_hr = hr_gt.shape
            if hasattr(model, 'set_image_size'):
                model.set_image_size((h_hr, w_hr))

            # Sampling
            start_time = time.time()
            z = model.prior.sample(lr.shape[0], device=device)
            sr = model(z, cond=lr, mode='generate')
            
            # Ensure output size matches GT
            if sr.shape != hr_gt.shape:
                sr = FV.resize(sr, [h_hr, w_hr], interpolation=FV.InterpolationMode.BICUBIC)
                
            generation_time = time.time() - start_time

            # Clamp results
            sr = torch.clamp(sr, 0, 1)
            
            # Check for adaptive steps usage
            steps_used = 200
            if hasattr(model.model.diffusion, 'get_actual_steps_used'):
                actual = model.model.diffusion.get_actual_steps_used()
                if actual: steps_used = actual

            results.append({
                'lr': lr.cpu(),
                'sr': sr.cpu(),
                'hr': hr_gt.cpu(),
                'time': generation_time,
                'steps': steps_used
            })
            
            # Optional: Save images logic can be added here or in visualization utils

    return results
