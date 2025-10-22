# data_loader.py
"""
This module provides functions for loading and preprocessing datasets
(MNIST and SVHN) for the domain adaptation experiments. It includes
functionality to create low-resource subsets and traditionally augmented
subsets of the SVHN dataset, as well as combining real and GAN-generated data.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import numpy as np
import os
from typing import Tuple


class ToTensorLong:
    def __call__(self, y):
        return torch.tensor(y, dtype=torch.long)


def get_dataloaders(
    low_resource_size: int, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    target_transform = ToTensorLong()
    """
    Prepares and returns data loaders for MNIST (source domain) and SVHN (target domain) datasets.
    This function handles downloading, transforming, and creating various subsets of the data,
    including a low-resource subset of SVHN and a traditionally augmented version of it.

    Args:
        low_resource_size (int): The total number of samples to use for the low-resource SVHN subset.
                                 These samples are distributed as evenly as possible across classes.
                                 A value of 0 means no low-resource subset is created (though
                                 the loader will still be returned, potentially empty).
        batch_size (int): The batch size to be used for all generated data loaders.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, DataLoader]: A tuple containing:
            - mnist_train_loader (DataLoader): DataLoader for the full MNIST training data.
            - svhn_low_resource_loader (DataLoader): DataLoader for the low-resource SVHN training data.
            - svhn_trad_aug_loader (DataLoader): DataLoader for the traditionally augmented
                                                 low-resource SVHN training data.
            - svhn_test_loader (DataLoader): DataLoader for the full SVHN test data.
    """
    # Define a common image transformation pipeline for both MNIST and SVHN.
    # Images are resized to 32x32, converted to grayscale (1 channel),
    # transformed to PyTorch tensors, and normalized to [-1, 1].
    transform = transforms.Compose(
        [
            transforms.Resize(32),                      # Resize images to 32x32 pixels
            transforms.Grayscale(num_output_channels=1), # Convert to grayscale (1 channel)
            transforms.ToTensor(),                      # Convert PIL Image to PyTorch Tensor
            transforms.Normalize((0.5,), (0.5,)),       # Normalize pixel values to [-1, 1]
        ]
    )

    # A transform to ensure integer labels are converted to LongTensors,
    # which is required for PyTorch's CrossEntropyLoss.

    # --- Source Domain: MNIST Dataset ---
    # Load the MNIST training dataset. If not present, it will be downloaded.
    mnist_train = datasets.MNIST(
        root="./data",                      # Directory where data is stored/downloaded
        train=True,                         # Specifies training set
        download=True,                      # Downloads the dataset if not found
        transform=transform,                # Apply image transformations
        target_transform=target_transform,  # Apply label transformations
    )
    # Create a DataLoader for MNIST training data.
    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2)

    # --- Target Domain: SVHN Dataset ---
    # Load the SVHN training dataset.
    svhn_train = datasets.SVHN(
        root="./data",
        split="train",
        download=True,
        transform=transform,
        target_transform=target_transform,
    )
    # Load the SVHN test dataset.
    svhn_test = datasets.SVHN(
        root="./data",
        split="test",
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    # Create a DataLoader for SVHN test data. Shuffling is not necessary for evaluation.
    svhn_test_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2)

    # --- Create the Low-Resource Target Set (SVHN) ---
    # This subset is used to simulate a scenario with limited labeled target data.
    targets = np.array(svhn_train.labels)
    indices = []
    # Calculate samples per class, ensuring at least 1 sample if low_resource_size > 0
    samples_per_class = max(1, int(low_resource_size / 10)) if low_resource_size > 0 else 0

    if low_resource_size > 0:
        for i in range(10): # Iterate through each of the 10 classes
            class_indices = np.where(targets == i)[0] # Get indices for current class
            # Sample 'num_to_sample' from available indices without replacement
            num_to_sample = min(samples_per_class, len(class_indices))
            indices.extend(np.random.choice(class_indices, num_to_sample, replace=False))

    # Create a Subset using the selected indices.
    svhn_low_resource_subset = Subset(svhn_train, indices)
    # Create a DataLoader for the low-resource SVHN training data.
    svhn_low_resource_loader = DataLoader(
        svhn_low_resource_subset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2
    )

    # --- Create the Traditionally Augmented Target Set (SVHN) ---
    # This subset applies traditional image augmentations to the low-resource data
    # to improve classifier generalization.
    transform_augmented = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Random affine transformations
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    # Create a new SVHN dataset instance with the augmentation transform.
    svhn_train_augmented = datasets.SVHN(
        root="./data",
        split="train",
        download=True,
        transform=transform_augmented,
        target_transform=target_transform,
    )
    # Create a Subset using the same indices as the low-resource set.
    svhn_augmented_subset = Subset(svhn_train_augmented, indices)
    # Create a DataLoader for the traditionally augmented SVHN training data.
    svhn_trad_aug_loader = DataLoader(
        svhn_augmented_subset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2
    )

    return (
        mnist_train_loader,
        svhn_low_resource_loader,
        svhn_trad_aug_loader,
        svhn_test_loader,
    )


def get_gan_augmented_loader(
    svhn_low_resource_loader: DataLoader,
    generated_images: torch.Tensor,
    generated_labels: torch.Tensor,
    batch_size: int = 64,
) -> DataLoader:
    """
    Combines real low-resource data with GAN-generated data into a single DataLoader.
    This is used for the GAN augmentation scenario, where synthetic images
    are used to expand the limited target domain training set.

    Args:
        svhn_low_resource_loader (DataLoader): DataLoader for the real low-resource SVHN data.
        generated_images (torch.Tensor): Tensor containing the GAN-generated images.
                                         Expected shape: (N, C, H, W).
        generated_labels (torch.Tensor): Tensor containing the class labels corresponding
                                         to the generated images. Expected shape: (N,).
        batch_size (int): The batch size for the combined data loader.

    Returns:
        DataLoader: A DataLoader containing both real low-resource and GAN-generated data,
                    shuffled for effective training.
    """
    # Ensure generated_labels is of type torch.long, as required for classification loss functions.
    generated_labels = generated_labels.to(dtype=torch.long)
    # Create a TensorDataset from the generated images and labels.
    generated_dataset = TensorDataset(generated_images, generated_labels)

    # Concatenate the real low-resource dataset with the generated dataset.
    # The .dataset attribute is used to get the underlying Dataset object from the DataLoader.
    combined_dataset = ConcatDataset(
        [svhn_low_resource_loader.dataset, generated_dataset]
    )
    # Create a DataLoader for the combined dataset.
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2)

    return combined_loader
