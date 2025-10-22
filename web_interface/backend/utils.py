# utils.py
"""
This module contains utility functions used across the project,
primarily for saving and visualizing GAN-generated images.
"""
import torch
import torchvision.utils as vutils
import os


def save_gan_images(images: torch.Tensor, epoch: int, n_classes: int = 10) -> None:
    """
    Saves a grid of GAN-generated images to a specified directory.
    The images are arranged in a grid, typically with each row representing a class,
    to visualize the generator's progress over training epochs.

    Args:
        images (torch.Tensor): A batch of images to save. Expected shape (N, C, H, W),
                               where N is the number of images, C is channels, H is height,
                               and W is width. Pixel values are expected to be in [-1, 1].
        epoch (int): The current epoch number, used for naming the output file
                     (e.g., 'epoch_001.png').
        n_classes (int): The number of classes. This is used to determine the number
                         of images per row in the grid, typically arranging images
                         such that each row corresponds to a specific class. Defaults to 10.
    """
    # Create the output directory if it doesn't already exist.
    os.makedirs("outputs/gan_images", exist_ok=True)
    # Create a grid of images. `nrow` specifies the number of images per row.
    # `normalize=True` scales image pixel values to [0, 1] for proper visualization.
    grid = vutils.make_grid(images, nrow=n_classes, normalize=True)
    # Save the grid image to the specified path, formatted with the epoch number.
    vutils.save_image(grid, f"outputs/gan_images/epoch_{epoch:03d}.png")
