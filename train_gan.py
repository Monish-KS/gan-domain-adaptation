# train_gan.py
"""
This module provides functionality for training a Conditional Deep Convolutional
Generative Adversarial Network (DCGAN). It includes the main training loop
for both the Generator and Discriminator, handles data loading, model
initialization, and saving of generated images and model checkpoints.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter for TensorBoard logging
import datetime # Import datetime for unique log directory names
import torchvision.utils as vutils # Import for logging image grids to TensorBoard

from models import Generator, Discriminator, weights_init
from data_loader import get_dataloaders
from utils import save_gan_images


def train_gan(
    n_samples: int,
    num_epochs: int = 100,
    nz: int = 100,
    lr: float = 0.0002,
    beta1: float = 0.5,
    batch_size: int = 64,
    ngf: int = 64,
    ndf: int = 64,
    nc: int = 1,
    n_classes: int = 10,
    output_dir: str = "outputs", # Added output_dir parameter with default
) -> None:
    """
    Trains a Conditional Deep Convolutional Generative Adversarial Network (DCGAN)
    on a low-resource subset of the SVHN dataset. The GAN consists of a Generator
    (netG) and a Discriminator (netD), which are trained adversarially.

    Args:
        n_samples (int): The total number of low-resource SVHN samples to use for GAN training.
                         These samples are distributed as evenly as possible across classes.
        num_epochs (int): The total number of training epochs for the GAN. Defaults to 100.
        nz (int): Size of the latent vector (noise) input to the Generator. Defaults to 100.
        lr (float): Learning rate for the Adam optimizers of both Generator and Discriminator.
                    Defaults to 0.0002.
        beta1 (float): Beta1 hyperparameter for the Adam optimizers. Defaults to 0.5.
        batch_size (int): Batch size for the data loader during GAN training. Defaults to 64.
        ngf (int): Size of feature maps in the Generator. This scales the complexity of G.
                   Defaults to 64.
        ndf (int): Size of feature maps in the Discriminator. This scales the complexity of D.
                   Defaults to 64.
        nc (int): Number of channels in the input/output images (e.g., 1 for grayscale).
                  Defaults to 1.
        n_classes (int): Number of classes for conditional generation/discrimination.
                         Defaults to 10 (for digits 0-9).
        output_dir (str): Base directory to save all GAN training outputs (TensorBoard logs,
                          generated images, model checkpoints). Defaults to "outputs".
    """
    # Determine the device to run the training on (GPU if available, otherwise CPU).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting GAN training on device: {device}")

    # --- Setup TensorBoard Logger ---
    # Create a unique log directory for each GAN training run, nested within the provided output_dir.
    log_dir = os.path.join(
        output_dir, "tensorboard_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_GAN_N{n_samples}"
    )
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs for GAN training will be saved to: {log_dir}")

    # Retrieve only the low-resource SVHN data loader, as the GAN is trained
    # to generate samples similar to this limited target data.
    _, svhn_low_resource_loader, _, _ = get_dataloaders(
        low_resource_size=n_samples, batch_size=batch_size
    )
    if len(svhn_low_resource_loader.dataset) == 0:
        print(f"Warning: No low-resource SVHN samples available for n_samples={n_samples}. GAN training skipped.")
        return # Exit if no data to train on

    # --- Model Initialization ---
    # Create the Generator and Discriminator networks and move them to the specified device.
    netG = Generator(nz=nz, ngf=ngf, nc=nc, n_classes=n_classes).to(device)
    netD = Discriminator(nc=nc, ndf=ndf, n_classes=n_classes).to(device)

    # Apply custom weights initialization to all convolutional and batch normalization layers.
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Initialize Binary Cross-Entropy Loss function, commonly used for GANs.
    criterion = nn.BCELoss()

    # --- Fixed Noise and Labels for Visualization ---
    # These are used to periodically generate and save images during training,
    # allowing for visual tracking of the Generator's progress.
    fixed_noise = torch.randn(100, nz, device=device) # 100 random noise vectors
    # Create 10 samples for each of the 10 classes (0-9)
    fixed_labels = torch.arange(0, n_classes, device=device).repeat(10)

    # Log a grid of real images to TensorBoard for comparison
    real_batch = next(iter(svhn_low_resource_loader))
    writer.add_image("GAN/Real Images", vutils.make_grid(real_batch[0][:100], nrow=10, normalize=True), 0)

    # --- Optimizers ---
    # Initialize Adam optimizers for both Generator and Discriminator.
    # Beta1 is a hyperparameter for Adam, typically set to 0.5 for GANs.
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print(f"Starting GAN training with {num_epochs} epochs on {n_samples} SVHN samples...")
    # --- GAN Training Loop ---
    for epoch in range(num_epochs):
        for i, (real_images, real_labels) in enumerate(svhn_low_resource_loader):
            # Move real images and labels to the appropriate device
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            b_size = real_images.size(0) # Current batch size

            # ============================================
            # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
            # ============================================
            netD.zero_grad() # Zero gradients for discriminator

            # Train with all-real batch
            # Create a tensor of real labels (all ones)
            label_real = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            # Forward pass real batch through Discriminator
            output = netD(real_images, real_labels).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label_real)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            # Train with all-fake batch
            # Generate batch of latent vectors (noise)
            noise = torch.randn(b_size, nz, device=device)
            # Generate fake labels for conditional generation
            fake_labels = torch.randint(0, n_classes, (b_size,), device=device)
            # Generate fake images with the Generator
            fake_images = netG(noise, fake_labels)
            # Create a tensor of fake labels (all zeros)
            label_fake = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
            # Classify all fake batch with Discriminator. Detach fake_images from
            # the generator's graph to prevent gradients from flowing back to G.
            output = netD(fake_images.detach(), fake_labels).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label_fake)
            # Calculate gradients for D in backward pass
            errD_fake.backward()
            # Add the gradients from the real and fake batches
            errD = errD_real + errD_fake
            # Update Discriminator
            optimizerD.step()

            # ============================================
            # (2) Update Generator network: maximize log(D(G(z)))
            # ============================================
            netG.zero_grad() # Zero gradients for generator
            # Since we just updated D, perform another forward pass of all-fake batch through D.
            # This time, we want D to classify the fakes as real (label_real).
            output = netD(fake_images, fake_labels).view(-1)
            # Calculate G's loss based on D's output
            errG = criterion(
                output, label_real
            ) # Generator wants discriminator to think fakes are real
            # Calculate gradients for G
            errG.backward()
            # Update Generator
            optimizerG.step()

        # Print training statistics for the current epoch
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss_D: {errD.item():.4f} "
            f"Loss_G: {errG.item():.4f}"
        )
        # Log losses to TensorBoard
        writer.add_scalar("GAN/Loss_D", errD.item(), epoch)
        writer.add_scalar("GAN/Loss_G", errG.item(), epoch)

        # Save generated images and model checkpoint periodically
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            with torch.no_grad(): # Disable gradient calculations for inference
                # Generate a batch of fake images using the fixed noise and labels
                fake = netG(fixed_noise, fixed_labels).detach().cpu()
            # Save the generated images to a file within the experiment's output directory.
            gan_images_dir = os.path.join(output_dir, "gan_images")
            os.makedirs(gan_images_dir, exist_ok=True)
            save_gan_images(fake, epoch + 1, n_classes=n_classes, output_dir=gan_images_dir)
            print(f"Saved generated images for epoch {epoch+1}.")
            # Log generated images to TensorBoard
            img_grid = vutils.make_grid(fake, nrow=n_classes, normalize=True)
            writer.add_image("GAN/Generated Images", img_grid, epoch)

            # --- Research Paper Enhancement: GAN Evaluation Metrics (FID/Inception Score) ---
            # For a research paper, it is crucial to evaluate GAN performance quantitatively
            # using metrics like Fréchet Inception Distance (FID) or Inception Score (IS).
            # These metrics require a pre-trained InceptionV3 model to extract features
            # from real and generated images.
            #
            # Implementation Notes:
            # 1. Install a library like `pytorch-fid` or `clean-fid`.
            #    (e.g., `pip install pytorch-fid`)
            # 2. Load real images (e.g., from svhn_low_resource_loader or a larger SVHN set).
            # 3. Generate a larger batch of synthetic images (e.g., 10,000 to 50,000).
            # 4. Calculate FID/IS between real and generated image sets.
            # 5. Log the FID/IS scores to TensorBoard (e.g., `writer.add_scalar("GAN/FID", fid_score, epoch)`).
            #
            # Example (conceptual, requires actual FID library integration):
            # if (epoch + 1) % 50 == 0: # Calculate FID every 50 epochs
            #     # Generate a large number of synthetic images for FID calculation
            #     num_fid_samples = 10000
            #     fid_noise = torch.randn(num_fid_samples, nz, device=device)
            #     fid_labels = torch.randint(0, n_classes, (num_fid_samples,), device=device)
            #     with torch.no_grad():
            #         fid_fake_images = netG(fid_noise, fid_labels).cpu()
            #
            #     # Assuming a function `calculate_fid` exists from an external library
            #     # fid_score = calculate_fid(real_images_for_fid, fid_fake_images)
            #     # writer.add_scalar("GAN/FID", fid_score, epoch)
            #     # print(f"Epoch {epoch+1}: FID Score = {fid_score:.2f}")
            #
            # This section is a placeholder to highlight where such an integration would occur.
            # For actual research, this would be fully implemented.

    # --- Save the Final Trained Generator Model ---
    # Ensure the checkpoint directory exists before saving.
    gan_checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(gan_checkpoints_dir, exist_ok=True)
    # Save the state dictionary of the Generator model.
    gan_model_path = os.path.join(gan_checkpoints_dir, f"gan_generator_n{n_samples}.pth")
    torch.save(netG.state_dict(), gan_model_path)
    print(f"Finished GAN Training. Generator model saved to {gan_model_path}")

    # Close the TensorBoard writer
    writer.close()
    print(f"TensorBoard writer closed. View logs with: tensorboard --logdir {os.path.dirname(log_dir)}")


if __name__ == "__main__":
    # --- Argument Parsing ---
    # Set up command-line argument parsing for configurable experiment parameters.
    parser = argparse.ArgumentParser(description="Train a Conditional DCGAN.")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of low-resource samples for the target domain (SVHN) to train the GAN on. Must be non-negative.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs for the GAN. Must be positive. Defaults to 100.",
    )
    parser.add_argument(
        "--nz",
        type=int,
        default=100,
        help="Size of the latent vector (noise) input to the Generator. Must be positive. Defaults to 100.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate for Adam optimizers. Must be positive. Defaults to 0.0002.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.5,
        help="Beta1 hyperparameter for Adam optimizers. Must be between 0 and 1. Defaults to 0.5.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data loaders during GAN training. Must be positive. Defaults to 64.",
    )
    parser.add_argument(
        "--ngf",
        type=int,
        default=64,
        help="Size of feature maps in the Generator. Must be positive. Defaults to 64.",
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=64,
        help="Size of feature maps in the Discriminator. Must be positive. Defaults to 64.",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=1,
        help="Number of channels in input/output images (e.g., 1 for grayscale). Must be positive. Defaults to 1.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=10,
        help="Number of classes for conditional generation/discrimination. Must be positive. Defaults to 10.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base directory to save all GAN training outputs (TensorBoard logs, generated images, model checkpoints).",
    )
    args = parser.parse_args()

    # --- Input Validation ---
    if args.n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer.")
    if args.epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if args.nz <= 0:
        raise ValueError("nz (latent vector size) must be a positive integer.")
    if args.lr <= 0:
        raise ValueError("lr (learning rate) must be a positive float.")
    if not (0 <= args.beta1 <= 1):
        raise ValueError("beta1 must be between 0 and 1.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if args.ngf <= 0:
        raise ValueError("ngf (generator feature map size) must be a positive integer.")
    if args.ndf <= 0:
        raise ValueError("ndf (discriminator feature map size) must be a positive integer.")
    if args.nc <= 0:
        raise ValueError("nc (number of channels) must be a positive integer.")
    if args.n_classes <= 0:
        raise ValueError("n_classes (number of classes) must be a positive integer.")

    # Call the main GAN training function with parsed arguments.
    train_gan(
        n_samples=args.n_samples,
        num_epochs=args.epochs,
        nz=args.nz,
        lr=args.lr,
        beta1=args.beta1,
        batch_size=args.batch_size,
        ngf=args.ngf,
        ndf=args.ndf,
        nc=args.nc,
        n_classes=args.n_classes,
        output_dir=args.output_dir, # Pass output_dir to train_gan function
    )
