# main_experiment.py
"""
This module orchestrates the domain adaptation experiments, including
classifier training, evaluation, and various domain adaptation scenarios
(source-only, fine-tuning, traditional augmentation, and GAN-based augmentation).
It handles data loading, model initialization, pre-training, and result reporting.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader # Explicitly import DataLoader for type hinting
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter for TensorBoard logging
import datetime # Import datetime for unique log directory names
import json # Import for saving experiment results to JSON

from models import Classifier, Generator
from data_loader import get_dataloaders, get_gan_augmented_loader


# --- Training and Evaluation Functions for Classifier ---
def train_classifier(
    model: nn.Module,
    train_loader: 'DataLoader', # Use string literal for forward reference
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    current_epoch: int, # Added current_epoch
    total_epochs: int, # Added total_epochs
) -> None:
    """
    Trains the classifier model for one epoch.

    Args:
        model (nn.Module): The classifier model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run the training on (cpu or cuda).
        current_epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs.
    """
    model.train() # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # Zero the gradients before running the backward pass
        output = model(data)
        loss = criterion(output, target)
        loss.backward() # Compute gradient of the loss with respect to model parameters
        optimizer.step() # Perform a single optimization step (parameter update)
        if batch_idx % 100 == 0: # Print loss every 100 batches
            print(f"  Epoch {current_epoch}/{total_epochs} Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")


def evaluate_classifier(
    model: nn.Module, test_loader: 'DataLoader', device: torch.device
) -> tuple[float, str]:
    """
    Evaluates the classifier model on the test set.

    Args:
        model (torch.nn.Module): The classifier model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        device (torch.device): Device to run the evaluation on (cpu or cuda).

    Returns:
        tuple[float, str]: A tuple containing:
            - accuracy (float): The accuracy score on the test set.
            - report (str): A classification report string.
    """
    model.eval() # Set the model to evaluation mode
    all_preds = []
    all_targets = []
    with torch.no_grad(): # Disable gradient calculations during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1) # Get the index of the max log-probability
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, digits=4)
    return accuracy, report


def _pretrain_classifier(
    classifier: nn.Module,
    mnist_loader: 'DataLoader',
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    output_dir: str, # Added output_dir parameter
) -> nn.Module:
    """
    Handles the pre-training of the classifier on the MNIST dataset.

    Args:
        classifier (nn.Module): The classifier model to pre-train.
        mnist_loader (DataLoader): DataLoader for the MNIST training data.
        optimizer (optim.Optimizer): Optimizer for the classifier.
        criterion (nn.Module): Loss function for the classifier.
        device (torch.device): Device to run training on.
        epochs (int): Number of epochs to pre-train.
        output_dir (str): Base directory for saving outputs.

    Returns:
        nn.Module: The pre-trained classifier model.
    """
    mnist_model_path = os.path.join(output_dir, "checkpoints", "classifier_mnist_pretrained.pth")

    if not os.path.exists(mnist_model_path):
        print(f"Pre-training classifier on MNIST for {epochs} epochs...")
        for epoch in range(epochs):
            train_classifier(classifier, mnist_loader, optimizer, criterion, device, epoch + 1, epochs)
            print(f"MNIST Pre-train Epoch {epoch+1}/{epochs} complete.")
        os.makedirs(os.path.dirname(mnist_model_path), exist_ok=True)
        torch.save(classifier.state_dict(), mnist_model_path)
        print(f"Pre-trained MNIST classifier saved to {mnist_model_path}")
    else:
        print(f"Loading pre-trained MNIST classifier from {mnist_model_path}...")
        classifier.load_state_dict(torch.load(mnist_model_path, map_location=device))
        print("Pre-trained MNIST classifier loaded.")
    return classifier


def _run_source_only_scenario(
    classifier: nn.Module, device: torch.device, output_dir: str # Added output_dir
) -> nn.Module:
    """
    Handles the "source_only" scenario: evaluates the MNIST pre-trained classifier
    directly on the SVHN test set without any fine-tuning.

    Args:
        classifier (nn.Module): The MNIST pre-trained classifier.
        device (torch.device): Device to run evaluation on.
        output_dir (str): Base directory for saving outputs.

    Returns:
        nn.Module: The classifier (unchanged).
    """
    print("Scenario: Source Only - Evaluating MNIST pre-trained model directly on SVHN.")
    model_save_path = os.path.join(output_dir, "checkpoints", "classifier_source_only.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(classifier.state_dict(), model_save_path)
    print(f"Source-only classifier model saved to {model_save_path}")
    return classifier


def _run_fine_tune_scenario(
    classifier: nn.Module,
    svhn_low_loader: 'DataLoader',
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    output_dir: str, # Added output_dir
) -> nn.Module:
    """
    Handles the "fine_tune" scenario: fine-tunes the classifier on the low-resource
    SVHN dataset.

    Args:
        classifier (nn.Module): The classifier to fine-tune.
        svhn_low_loader (DataLoader): DataLoader for the low-resource SVHN training data.
        optimizer (torch.optim.Optimizer): Optimizer for the classifier.
        criterion (torch.nn.Module): Loss function for the classifier.
        device (torch.device): Device to run training on.
        epochs (int): Number of epochs to fine-tune.
        output_dir (str): Base directory for saving outputs.

    Returns:
        nn.Module: The fine-tuned classifier model.
    """
    print(f"Scenario: Fine-tuning on low-resource SVHN data for {epochs} epochs...")
    for epoch in range(epochs):
        train_classifier(classifier, svhn_low_loader, optimizer, criterion, device, epoch + 1, epochs)
        print(f"SVHN Fine-tune Epoch {epoch+1}/{epochs} complete.")
    model_save_path = os.path.join(output_dir, "checkpoints", "classifier_fine_tune.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(classifier.state_dict(), model_save_path)
    print(f"Fine-tuned classifier model saved to {model_save_path}")
    return classifier


def _run_traditional_aug_scenario(
    classifier: nn.Module,
    svhn_trad_aug_loader: 'DataLoader',
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    output_dir: str, # Added output_dir
) -> nn.Module:
    """
    Handles the "traditional_aug" scenario: fine-tunes the classifier on traditionally
    augmented low-resource SVHN data.

    Args:
        classifier (nn.Module): The classifier to fine-tune.
        svhn_trad_aug_loader (DataLoader): DataLoader for the traditionally augmented
                                           SVHN training data.
        optimizer (torch.optim.Optimizer): Optimizer for the classifier.
        criterion (torch.nn.Module): Loss function for the classifier.
        device (torch.device): Device to run training on.
        epochs (int): Number of epochs to fine-tune.
        output_dir (str): Base directory for saving outputs.

    Returns:
        nn.Module: The fine-tuned classifier model.
    """
    print(f"Scenario: Fine-tuning on traditionally augmented low-resource SVHN data for {epochs} epochs...")
    for epoch in range(epochs):
        train_classifier(classifier, svhn_trad_aug_loader, optimizer, criterion, device, epoch + 1, epochs)
        print(f"SVHN Traditional Aug Epoch {epoch+1}/{epochs} complete.")
    model_save_path = os.path.join(output_dir, "checkpoints", "classifier_traditional_aug.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(classifier.state_dict(), model_save_path)
    print(f"Traditional augmented classifier model saved to {model_save_path}")
    return classifier


def _run_gan_aug_scenario(
    classifier: nn.Module,
    svhn_low_loader: 'DataLoader',
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: str, # Added output_dir
    gan_model_base_dir: str = None, # Added gan_model_base_dir
) -> nn.Module:
    """
    Handles the "gan_aug" scenario: fine-tunes the classifier on GAN-augmented
    low-resource SVHN data.

    Args:
        classifier (nn.Module): The classifier to fine-tune.
        svhn_low_loader (DataLoader): DataLoader for the real low-resource SVHN data.
        optimizer (torch.optim.Optimizer): Optimizer for the classifier.
        criterion (torch.nn.Module): Loss function for the classifier.
        device (torch.device): Device to run training on.
        args (argparse.Namespace): Experiment configuration arguments.
        output_dir (str): Base directory for saving outputs.
        gan_model_base_dir (str, optional): Base directory to load the GAN generator model from.
                                            If None, `output_dir` is used. Defaults to None.

    Returns:
        nn.Module: The fine-tuned classifier model.
    """
    print(f"  _run_gan_aug_scenario: output_dir={output_dir}, gan_model_base_dir={gan_model_base_dir}")
    print(f"Scenario: Fine-tuning on GAN-augmented SVHN data for {args.classifier_epochs_finetune} epochs...")
    # Load the pre-trained GAN generator model.
    generator = Generator(nz=args.gan_nz, ngf=args.gan_ngf, nc=args.gan_nc, n_classes=args.gan_n_classes).to(device)
    
    # Determine the directory from which to load the GAN model
    gan_load_dir = gan_model_base_dir if gan_model_base_dir else output_dir
    gan_model_path = os.path.join(gan_load_dir, "checkpoints", f"gan_generator_n{args.n_samples}.pth")
    print(f"  _run_gan_aug_scenario: gan_load_dir={gan_load_dir}, gan_model_path={gan_model_path}")
    
    if not os.path.exists(gan_model_path):
        raise FileNotFoundError(
            f"GAN generator model for n_samples={args.n_samples} not found at {gan_model_path}. "
            f"Please ensure the GAN was trained and saved to the correct output directory."
        )
    generator.load_state_dict(torch.load(gan_model_path, map_location=device))
    generator.eval() # Set generator to evaluation mode for inference

    # Generate synthetic data using the trained GAN generator.
    print(f"Generating {args.n_synthetic} synthetic images using GAN...")
    noise = torch.randn(args.n_synthetic, args.gan_nz, device=device)
    labels = torch.randint(0, args.gan_n_classes, (args.n_synthetic,), device=device)
    with torch.no_grad():
        synthetic_images = generator(noise, labels).cpu() # Move to CPU for DataLoader
    print("Synthetic images generated.")

    # Create a combined data loader with real low-resource SVHN data and GAN-generated data.
    gan_augmented_loader = get_gan_augmented_loader(
        svhn_low_loader, synthetic_images, labels.cpu(), batch_size=args.batch_size
    )
    print("GAN-augmented data loader created.")

    for epoch in range(args.classifier_epochs_finetune):
        train_classifier(
            classifier, gan_augmented_loader, optimizer, criterion, device, epoch + 1, args.classifier_epochs_finetune
        )
        print(f"SVHN GAN Aug Epoch {epoch+1}/{args.classifier_epochs_finetune} complete.")
    model_save_path = os.path.join(output_dir, "checkpoints", "classifier_gan_aug.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(classifier.state_dict(), model_save_path)
    print(f"GAN-augmented classifier model saved to {model_save_path}")
    return classifier


def run_experiment(
    args: argparse.Namespace,
    gan_model_base_dir: str = None,
) -> None:
    """
    Orchestrates and runs a domain adaptation experiment based on the specified scenario
    and configuration provided in the `args` namespace. This function sets up data loaders,
    initializes and pre-trains a classifier, and then executes one of several domain
    adaptation strategies (source_only, fine_tune, traditional_aug, gan_aug) before
    evaluating the final classifier.

    Args:
        args (argparse.Namespace): An object containing all experiment configuration
                                   parameters as attributes.
        gan_model_base_dir (str, optional): Base directory to load the GAN generator model from
                                            for the 'gan_aug' scenario. If None, `args.output_dir` is used.
                                            Defaults to None.
    """
    print(f"run_experiment: args.output_dir={args.output_dir}, gan_model_base_dir={gan_model_base_dir}")
    # --- Setup TensorBoard Logger ---
    # Create a unique log directory for each experiment run using a timestamp.
    # The log directory will be nested within the provided output_dir.
    log_dir = os.path.join(
        args.output_dir, "tensorboard_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{args.scenario}_N{args.n_samples}"
    )
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # --- Step 1: Device Configuration ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n===== RUNNING SCENARIO: {args.scenario.upper()} with N={args.n_samples} on {device} =====")

    # --- Step 2: Data Loading ---
    # Retrieve data loaders for MNIST (source) and various SVHN (target) configurations.
    mnist_loader, svhn_low_loader, svhn_trad_aug_loader, svhn_test_loader = (
        get_dataloaders(low_resource_size=args.n_samples, batch_size=args.batch_size)
    )

    # --- Step 3: Initialize Classifier ---
    classifier = Classifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)
    criterion = nn.CrossEntropyLoss()

    # --- Step 4: Pre-train Classifier on MNIST (Source Domain) ---
    # The classifier is pre-trained on the larger MNIST dataset to learn general features.
    classifier = _pretrain_classifier(
        classifier, mnist_loader, optimizer, criterion, device, args.classifier_epochs_mnist, args.output_dir
    )

    # --- Step 5: Execute the Selected Domain Adaptation Scenario ---
    # Each scenario involves different training strategies on the SVHN dataset.
    if args.scenario == "source_only":
        classifier = _run_source_only_scenario(classifier, device, args.output_dir)
    elif args.scenario == "fine_tune":
        classifier = _run_fine_tune_scenario(
            classifier, svhn_low_loader, optimizer, criterion, device, args.classifier_epochs_finetune, args.output_dir
        )
    elif args.scenario == "traditional_aug":
        classifier = _run_traditional_aug_scenario(
            classifier, svhn_trad_aug_loader, optimizer, criterion, device, args.classifier_epochs_finetune, args.output_dir
        )
    elif args.scenario == "gan_aug":
        classifier = _run_gan_aug_scenario(
            classifier, svhn_low_loader, optimizer, criterion, device, args, args.output_dir, gan_model_base_dir
        )

    # --- Step 6: Final Evaluation ---
    # Evaluate the trained/fine-tuned classifier on the full SVHN test set.
    print("\n--- Performing Final Evaluation on SVHN Test Set ---")
    accuracy, report = evaluate_classifier(classifier, svhn_test_loader, device)
    print(f"\n--- Results for Scenario: {args.scenario.upper()} (N={args.n_samples}) ---")
    print(f"Final Accuracy on SVHN Test Set: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)
    print("=" * 60)

    # Log final accuracy to TensorBoard
    writer.add_scalar("Final Accuracy/SVHN", accuracy, 0) # Log at step 0 for final result

    # --- Research Paper Enhancement: Ablation Study Capabilities ---
    # To conduct ablation studies, researchers typically vary one component or hyperparameter
    # at a time to understand its impact on the overall system performance.
    # With the current argparse setup, this can be done by running `main_experiment.py`
    # multiple times with different command-line arguments.
    #
    # Example:
    # 1. To study the effect of `n_samples`:
    #    `python main_experiment.py --scenario gan_aug --n_samples 10 --output_dir ./experiments/run_X`
    #    `python main_experiment.py --scenario gan_aug --n_samples 100 --output_dir ./experiments/run_Y`
    #    `python main_experiment.py --scenario gan_aug --n_samples 1000 --output_dir ./experiments/run_Z`
    #
    # 2. To compare different scenarios:
    #    `python main_experiment.py --scenario source_only --output_dir ./experiments/run_A`
    #    `python main_experiment.py --scenario fine_tune --output_dir ./experiments/run_B`
    #    `python main_experiment.py --scenario gan_aug --output_dir ./experiments/run_C`
    #
    # For systematic ablation studies, consider using a dedicated experiment management
    # tool like Sacred, MLflow, or Weights & Biases, which can automate tracking
    # configurations, metrics, and artifacts across many runs.

    # --- Save Experiment Results to File ---
    # Saving results to a structured file (e.g., JSON) is essential for
    # systematic analysis and comparison in research.
    results = {
        "scenario": args.scenario,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "classifier_lr": args.classifier_lr,
        "classifier_epochs_mnist": args.classifier_epochs_mnist,
        "classifier_epochs_finetune": args.classifier_epochs_finetune,
        "gan_nz": args.gan_nz,
        "gan_ngf": args.gan_ngf,
        "gan_ndf": args.gan_ndf,
        "gan_nc": args.gan_nc,
        "gan_n_classes": args.gan_n_classes,
        "n_synthetic": args.n_synthetic,
        "final_accuracy": accuracy,
        "classification_report": report, # Store as string or parse into dict if needed
        "timestamp": datetime.datetime.now().isoformat(),
        "log_dir": log_dir,
    }

    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_filename = f"{args.scenario}_N{args.n_samples}_results.json" # Simplified filename
    results_path = os.path.join(results_dir, results_filename)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Experiment results saved to: {results_path}")

    # Close the TensorBoard writer
    writer.close()
    print(f"TensorBoard writer closed. View logs with: tensorboard --logdir {os.path.dirname(log_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run domain adaptation experiments.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="gan_aug",
        choices=["source_only", "fine_tune", "traditional_aug", "gan_aug"],
        help="The experiment scenario to run.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of low-resource samples for the target domain (SVHN). Must be non-negative.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data loaders. Must be positive.",
    )
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the classifier's Adam optimizer. Must be positive.",
    )
    parser.add_argument(
        "--classifier_epochs_mnist",
        type=int,
        default=5,
        help="Number of epochs to pre-train the classifier on MNIST. Must be positive.",
    )
    parser.add_argument(
        "--classifier_epochs_finetune",
        type=int,
        default=20,
        help="Number of epochs to fine-tune the classifier on SVHN data. Must be positive.",
    )
    parser.add_argument(
        "--gan_nz",
        type=int,
        default=100,
        help="Size of the latent vector (noise) for the GAN generator. Must be positive.",
    )
    parser.add_argument(
        "--gan_ngf",
        type=int,
        default=64,
        help="Size of feature maps in the GAN generator. Must be positive.",
    )
    parser.add_argument(
        "--gan_ndf",
        type=int,
        default=64,
        help="Size of feature maps in the GAN discriminator. Must be positive.",
    )
    parser.add_argument(
        "--gan_nc",
        type=int,
        default=1,
        help="Number of channels in GAN input/output images. Must be positive.",
    )
    parser.add_argument(
        "--gan_n_classes",
        type=int,
        default=10,
        help="Number of classes for GAN conditioning. Must be positive.",
    )
    parser.add_argument(
        "--n_synthetic",
        type=int,
        default=5000,
        help="Number of synthetic images to generate for GAN augmentation. Must be non-negative.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory to save all experiment outputs (TensorBoard logs, results, models).",
    )
    parser.add_argument(
        "--gan_model_base_dir",
        type=str,
        default=None,
        help="Base directory to load the GAN generator model from for the 'gan_aug' scenario. If not provided, --output_dir is used.",
    )
    args = parser.parse_args()

    # --- Input Validation ---
    if args.n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if args.classifier_lr <= 0:
        raise ValueError("classifier_lr must be a positive float.")
    if args.classifier_epochs_mnist <= 0:
        raise ValueError("classifier_epochs_mnist must be a positive integer.")
    if args.classifier_epochs_finetune <= 0:
        raise ValueError("classifier_epochs_finetune must be a positive integer.")
    if args.gan_nz <= 0:
        raise ValueError("gan_nz must be a positive integer.")
    if args.gan_ngf <= 0:
        raise ValueError("gan_ngf must be a positive integer.")
    if args.gan_ndf <= 0:
        raise ValueError("gan_ndf must be a positive integer.")
    if args.gan_nc <= 0:
        raise ValueError("gan_nc must be a positive integer.")
    if args.gan_n_classes <= 0:
        raise ValueError("gan_n_classes must be a positive integer.")
    if args.n_synthetic < 0:
        raise ValueError("n_synthetic must be a non-negative integer.")

    run_experiment(args, gan_model_base_dir=args.gan_model_base_dir)
