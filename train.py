import argparse
import torch
from torch.utils.data import DataLoader
from src.models import ConditionalDensityModel
from src.data import DF2kDataset, ToyDataset, AugmentedDataset
from src.training import train_model

def main():
    parser = argparse.ArgumentParser(description="Train AdaSR Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_toy", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Dataset
    if args.use_toy:
        train_dataset = ToyDataset(num=1000, image_size=160)
    else:
        # Assuming DF2K structure
        train_dataset = DF2kDataset(args.data_dir, downscale_factor=4, train=True, random_crop_size=160)
        train_dataset = AugmentedDataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Model
    model = ConditionalDensityModel(input_channels=3, sr_factor=4)
    
    # Train
    train_model(
        model, 
        train_loader, 
        num_epochs=args.epochs, 
        lr=args.lr, 
        device=device,
        checkpoint_dir='checkpoints'
    )

if __name__ == "__main__":
    main()
