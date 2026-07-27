"""
Training script for UmbraLift.

This script implements a training pipeline for our Low Light Image Enhancement model while supporting multiple loss configurations, early stopping,
and various training parameters that can be configured through command line arguments.

Key features:
- Dynamic loss configuration
- Early stopping mechanism
- Gradient clipping
- Model checkpointing
- Multiple loss functions for comprehensive image enhancement
"""

import os
import argparse
import torch
import torch.optim
import model
import utils.losses as losses
import utils.dataloader as dataloader
from utils.device import get_device, describe

def weights_init(m):
    """Initialize network weights using normal distribution.
    
    Args:
        m: Network module
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
          
def train(config):
    """Main training function.
    
    Args:
        config: ArgumentParser object containing training configurations
    """
    device = get_device(config.device)
    print(f"Training on {describe(device)}")

    net = model.umbra_lift_net().to(device)

    net.apply(weights_init)
    if config.load_pretrain:
        net.load_state_dict(
            torch.load(config.pretrain_snapshot, map_location=device, weights_only=True))

    """
    Explanation of Weights:

    L_color (5.0): Set to a moderate weight to correct color discrepancies without dominating other losses.

    L_spa (1.5): Set higher weight to strongly enforce spatial consistency, crucial for maintaining sharpness and details.

    L_exp (10.0): Higher weight to adaptively enhance exposure based on input brightness, vital for varying low-light conditions.

    L_TV (200.0): Kept at a high weight to effectively suppress noise and artifacts, common in low-light images.

    L_contrast (5.0): Significant weight to improve visibility of details by enhancing local contrast.

    L_texture (3.0): Moderate weight to ensure texture details are preserved without introducing artifacts.
    """

    # Define the loss configurations with their respective weights
    loss_configs = [
    {
        "name": "low_light_enhancement",
        "weights": {
            "L_color": 5.0,          # Maintains color consistency
            "L_spa": 1.5,            # Preserves spatial structure
            "L_exp": 10.0,           # Enhances overall exposure
            "L_TV": 200.0,           # Reduces noise, smooths image
            "L_contrast": 5.0,       # Enhances local contrast
            "L_texture": 3.0,        # Preserves textures
        }
    }
]

    # Loop over each loss configuration
    for loss_config in loss_configs:
        print(f"Training with loss configuration: {loss_config['name']}")

        # Setup data loading
        train_dataset = dataloader.lowlight_loader(
            config.lowlight_images_path,
            size=config.image_size,
            max_images=config.max_train_images,
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=(device.type == "cuda"))

        # Initialize all loss functions. Those holding kernels are modules with
        # buffers, so they need moving to the compute device alongside the net.
        L_color = losses.L_color()
        L_spa = losses.L_spa().to(device)
        L_texture = losses.L_texture().to(device)
        L_exp = losses.L_exp(patch_size=16, mean_val=0.6)
        L_contrast = losses.L_contrast()
        L_TV = losses.L_TV()

        # Setup optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        net.train()

        # Initialize early stopping parameters
        best_loss = float('inf')
        patience = config.early_stopping_patience
        patience_counter = 0
        min_delta = 1e-4

        # Main training loop
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for iteration, img_lowlight in enumerate(train_loader):
                img_lowlight = img_lowlight.to(device)

                # Forward pass through the network
                enhanced_image, A = net(img_lowlight)

                # Calculate combined loss based on configuration
                loss = 0.0
                weights = loss_config["weights"]

                # Add individual loss components
                if "L_color" in weights:
                    loss += weights["L_color"] * torch.mean(L_color(enhanced_image))
                if "L_spa" in weights:
                    loss += weights["L_spa"] * torch.mean(L_spa(enhanced_image, img_lowlight))
                if "L_TV" in weights:
                    loss += weights["L_TV"] * torch.mean(L_TV(A))
                if "L_texture" in weights:
                    loss += weights["L_texture"] * torch.mean(L_texture(img_lowlight, enhanced_image))
                if "L_exp" in weights:
                    loss += weights["L_exp"] * torch.mean(L_exp(enhanced_image))
                if "L_contrast" in weights:
                    loss += weights["L_contrast"] * torch.mean(L_contrast(enhanced_image))

                # Handle NaN losses
                if torch.isnan(loss).any():
                    print("NaN detected in loss. Skipping iteration.")
                    continue

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Progress display
                if (iteration + 1) % config.display_iter == 0:
                    print(f"Epoch [{epoch+1}/{config.num_epochs}], Iteration [{iteration+1}], Loss: {loss.item()}")

                # Save model checkpoint
                if (iteration + 1) % config.checkpoint_iter == 0:
                    checkpoint_name = f"model-{loss_config['name']}-epoch{epoch}-iteration{iteration}.pth"
                    torch.save(net.state_dict(), os.path.join(config.checkpoints_folder, checkpoint_name))

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / num_batches

            # Early stopping logic. Note this tracks *training* loss -- there is
            # no held-out split, so it detects convergence, not overfitting.
            if avg_epoch_loss < best_loss - min_delta:
                best_loss = avg_epoch_loss
                patience_counter = 0
                # Save best model under the name every other script loads.
                best_path = os.path.join(config.snapshots_folder, config.best_model_name)
                torch.save(net.state_dict(), best_path)
                print(f"Epoch [{epoch+1}/{config.num_epochs}] avg loss {avg_epoch_loss:.4f} -- saved {best_path}")
            else:
                patience_counter += 1
                print(f"Epoch [{epoch+1}/{config.num_epochs}] avg loss {avg_epoch_loss:.4f} "
                      f"(no improvement, patience {patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"Finished training with loss configuration: {loss_config['name']}")


if __name__ == "__main__":
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description='UmbraLift Training Script')

    # Training configuration parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/", help='Path to low-light training images')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
    parser.add_argument('--grad_clip_norm', type=float, default=1, help='Gradient clipping norm')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Validation batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--display_iter', type=int, default=5, help='Display loss every N iterations')
    parser.add_argument('--checkpoint_iter', type=int, default=5, help='Save model every N iterations')
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/", help='Directory to save best model')
    parser.add_argument('--checkpoints_folder', type=str, default="checkpoints/", help='Directory to save model checkpoints')
    # store_true rather than type=bool: `--load_pretrain False` would otherwise
    # evaluate to True, since bool("False") is truthy.
    parser.add_argument('--load_pretrain', action='store_true', help='Load a pretrained model before training')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Number of epochs to wait before early stopping')
    parser.add_argument('--pretrain_snapshot', type=str, default= "snapshots/model-best.pth", help='Pretrained model snapshot')
    parser.add_argument('--best_model_name', type=str, default="model-best.pth", help='Filename for the best checkpoint')
    parser.add_argument('--image_size', type=int, default=128, help='Training crop size (must be divisible by 8)')
    parser.add_argument('--max_train_images', type=int, default=None, help='Cap on training images (default: use all)')
    parser.add_argument('--device', type=str, default=None, help='Force a device: cuda, mps or cpu')

    config = parser.parse_args()

    # Create snapshots and checkpoints directories if they don't exist
    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.checkpoints_folder, exist_ok=True)

    train(config)