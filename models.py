# models.py
"""
This module defines the neural network architectures used in the project,
including a Classifier for image classification and a Conditional DCGAN
(Generator and Discriminator) for synthetic image generation.
"""
import torch
import torch.nn as nn

# --- Helper function for GANs ---
def weights_init(m: nn.Module) -> None:
    """
    Initializes weights of convolutional and batch normalization layers.
    Applies a normal distribution for weights and sets bias to 0 for BatchNorm.

    Args:
        m (nn.Module): A PyTorch module (e.g., Conv2d, BatchNorm2d).
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Classifier(nn.Module):
    """
    A Convolutional Neural Network (CNN) classifier designed for 32x32 grayscale images.
    This architecture is suitable for datasets like MNIST and SVHN, performing
    feature extraction followed by classification.

    The network consists of:
    - Two convolutional layers with ReLU activation and Max Pooling.
    - A flattening layer to convert 2D feature maps into a 1D vector.
    - Two fully connected (linear) layers with ReLU activation and Dropout for regularization.
    - An output layer producing logits for 10 classes.
    """
    def __init__(self) -> None:
        """
        Initializes the Classifier network's layers.
        """
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, 5, 1, 2),  # Input: 1x32x32, Output: 32x32x32 (padding maintains size)
            nn.ReLU(True),             # In-place ReLU activation
            nn.MaxPool2d(2, 2),        # Output: 32x16x16 (downsamples by 2)

            # Second convolutional block
            nn.Conv2d(32, 64, 5, 1, 2), # Input: 32x16x16, Output: 64x16x16
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),        # Output: 64x8x8

            nn.Flatten(),              # Flatten feature maps into a vector (64 * 8 * 8 = 4096)

            # Fully connected layers
            nn.Linear(64 * 8 * 8, 1024), # First fully connected layer
            nn.ReLU(True),
            nn.Dropout(0.5),           # Dropout for regularization
            nn.Linear(1024, 10)        # Output layer for 10 classes
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classifier.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, 1, 32, 32).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 10) representing class logits.
        """
        return self.main(input)

class Generator(nn.Module):
    """
    A Conditional Deep Convolutional Generative Adversarial Network (DCGAN) Generator.
    This network takes a latent noise vector and a class label as input, and
    generates a 32x32 grayscale image. The generation process is conditioned
    on the provided class label.

    The architecture uses `ConvTranspose2d` layers for upsampling,
    Batch Normalization, and ReLU activations, with a Tanh activation
    at the output to scale pixel values to [-1, 1].
    """
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 1, n_classes: int = 10) -> None:
        """
        Initializes the Generator network's layers.

        Args:
            nz (int): Size of the latent vector (noise).
            ngf (int): Size of feature maps in the generator.
            nc (int): Number of channels in the output images (e.g., 1 for grayscale).
            n_classes (int): Number of classes for conditional generation.
        """
        super(Generator, self).__init__()
        # Embedding layer to project class labels into a dense vector space
        self.label_emb = nn.Embedding(n_classes, nz)

        # Main sequential block for image generation
        self.main = nn.Sequential(
            # Input: (noise + label_embedding) reshaped to (nz*2)x1x1
            # First ConvTranspose2d layer: (nz*2) -> (ngf*4)
            nn.ConvTranspose2d(nz * 2, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 4 x 4

            # Second ConvTranspose2d layer: (ngf*4) -> (ngf*2)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 8 x 8

            # Third ConvTranspose2d layer: (ngf*2) -> (ngf)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 16 x 16

            # Fourth ConvTranspose2d layer: (ngf) -> (nc)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh() # Tanh activation to scale output pixels to [-1, 1]
            # Final State size: (nc) x 32 x 32
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator.

        Args:
            noise (torch.Tensor): Latent noise vector of shape (batch_size, nz).
            labels (torch.Tensor): Class labels of shape (batch_size,).

        Returns:
            torch.Tensor: Generated images of shape (batch_size, nc, 32, 32).
        """
        # Concatenate label embedding and noise
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        x = x.view(x.size(0), -1, 1, 1) # Reshape for ConvTranspose2d
        return self.main(x)

class Discriminator(nn.Module):
    """
    A Conditional Deep Convolutional Generative Adversarial Network (DCGAN) Discriminator.
    This network takes a 32x32 grayscale image and a class label as input,
    and outputs a probability indicating whether the image is real or fake.
    The discrimination process is conditioned on the provided class label.

    The architecture uses `Conv2d` layers for downsampling, Batch Normalization,
    and LeakyReLU activations, with a Sigmoid activation at the output.
    """
    def __init__(self, nc: int = 1, ndf: int = 64, n_classes: int = 10) -> None:
        """
        Initializes the Discriminator network's layers.

        Args:
            nc (int): Number of channels in the input images (e.g., 1 for grayscale).
            ndf (int): Size of feature maps in the discriminator.
            n_classes (int): Number of classes for conditional discrimination.
        """
        super(Discriminator, self).__init__()
        # Embedding layer to project class labels into a vector that can be reshaped
        # and concatenated with the input image.
        self.label_embedding = nn.Embedding(n_classes, 32 * 32) # Embedding to match image spatial dimensions

        # Main sequential block for image discrimination
        self.main = nn.Sequential(
            # Input: (nc + 1) x 32 x 32 (concatenated image and reshaped label embedding)
            # First Conv2d layer: (nc + 1) -> (ndf)
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU activation

            # Second Conv2d layer: (ndf) -> (ndf*2)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Third Conv2d layer: (ndf*2) -> (ndf*4)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Fourth Conv2d layer (output layer): (ndf*4) -> 1
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() # Sigmoid activation to output a probability (real/fake)
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the discriminator.

        Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, nc, 32, 32).
            labels (torch.Tensor): Class labels of shape (batch_size,).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) representing real/fake probability.
        """
        # Concatenate label embedding (reshaped to image size) and image
        label_embedding_reshaped = self.label_embedding(labels).view(labels.size(0), 1, 32, 32)
        d_in = torch.cat((img, label_embedding_reshaped), 1)
        return self.main(d_in)