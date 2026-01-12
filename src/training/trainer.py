import torch
import torch.optim as optim
from tqdm import tqdm
import os

def train_model(model, train_loader, num_epochs=50, lr=1e-4, device='cuda', 
                save_checkpoints=True, checkpoint_dir='checkpoints', 
                checkpoint_interval=5, weight_decay=1e-5, warmup_epochs=2):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    model.train()
    
    # Optimizer setup as per original code
    optimizer = optim.Adam([
        {'params': model.model.diffusion.parameters(), 'weight_decay': weight_decay, 'base_lr': lr},
        {'params': model.model.lr_feats.parameters(), 'lr': 1e-5, 'weight_decay': weight_decay, 'base_lr': 1e-5},
    ], lr=lr, betas=(0.9, 0.999))

    # Scheduler
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=lr * 0.0001
    )

    best_loss = float('inf')
    loss_history = []

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Warmup
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                base_lr = param_group.get('base_lr', lr)
                param_group['lr'] = float(base_lr * warmup_factor)

        for batch in pbar:
            # Handle dictionary batch from our new dataset class
            hr = batch['image'].to(device)
            lr = batch['image_lr'].to(device)

            optimizer.zero_grad()
            # Forward pass with mode='loss'
            loss = model(hr, cond=lr, mode='loss')
            loss = loss.mean()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / (pbar.n + 1)})

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Update scheduler
        if epoch >= warmup_epochs:
            plateau_scheduler.step(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"✓ New best model saved! Loss: {best_loss:.4f}")

        # Periodic checkpoint
        if save_checkpoints and (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))

    return model
