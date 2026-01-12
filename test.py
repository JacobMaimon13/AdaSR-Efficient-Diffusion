import argparse
import torch
import os
from torch.utils.data import DataLoader
from src.models import ConditionalDensityModel
from src.data import DIV2KTestDataset, ToyDataset
from src.training import test_model
from src.utils import MetricCalculator, save_comparison_grid

def main():
    parser = argparse.ArgumentParser(description="Test AdaSR Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--adaptive", action="store_true", help="Enable Adaptive Decoding")
    parser.add_argument("--use_toy", action="store_true")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    if args.use_toy:
        test_dataset = ToyDataset(num=5, image_size=160)
    else:
        test_dataset = DIV2KTestDataset(args.data_dir) # Uses full images

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model Setup
    adaptive_config = None
    if args.adaptive:
        adaptive_config = {
            'complexity_method': 'gradient',
            'min_steps': 50,
            'max_steps': 200,
            'early_stopping_enabled': True
        }

    model = ConditionalDensityModel(
        input_channels=3, 
        sr_factor=4, 
        use_adaptive_decoding=args.adaptive,
        adaptive_decoding_config=adaptive_config
    )

    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded model from {args.model_path}")

    # Run Inference
    results = test_model(model, test_loader, device=device)

    # Calculate Metrics & Save
    metrics_calc = MetricCalculator(device)
    
    print("\n--- Evaluation Results ---")
    for i, res in enumerate(results):
        m = metrics_calc.calculate(res['sr'], res['hr'])
        m['steps'] = res['steps']
        m['time'] = res['time']
        
        print(f"Image {i}: PSNR={m.get('psnr',0):.2f} | Steps={m['steps']} | Time={m['time']:.2f}s")
        
        save_comparison_grid(
            res['lr'], res['sr'], res['hr'], 
            os.path.join(args.output_dir, f"result_{i}.png"), 
            metrics=m
        )

if __name__ == "__main__":
    main()
