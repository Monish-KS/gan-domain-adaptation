# run_all_experiments.py
"""
This script orchestrates a series of domain adaptation experiments,
running different scenarios and configurations. It manages output directories,
ensures proper logging, and saves all relevant results (TensorBoard logs,
JSON reports, trained models, generated images) in a structured, timestamped
manner for easy analysis and reproducibility.
"""
import argparse
import subprocess
import os
import datetime
import shutil

def run_all_experiments(args: argparse.Namespace) -> None:
    """
    Orchestrates and runs a series of domain adaptation experiments.

    Args:
        args (argparse.Namespace): Command-line arguments specifying experiment configurations.
    """
    base_output_dir = os.path.join("experiments", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"All experiment outputs will be saved under: {base_output_dir}")

    # Define a list of experiments to run. Each experiment is a dictionary
    # of arguments that will be passed to main_experiment.py.
    # For demonstration, we'll run a few scenarios with different n_samples.
    experiments_to_run = []

    # Example 1: Run GAN augmentation with varying n_samples
    for n_samples in [10, 100, 1000]:
        experiments_to_run.append({
            "scenario": "gan_aug",
            "n_samples": n_samples,
            "classifier_epochs_finetune": 20, # Use default or specify
            "gan_epochs": 100 # Ensure GAN is trained for enough epochs
        })

    # Example 2: Run all scenarios for a fixed n_samples
    fixed_n_samples = 100
    for scenario in ["source_only", "fine_tune", "traditional_aug", "gan_aug"]:
        if scenario == "gan_aug":
            # GAN aug is already covered above for varying n_samples, skip if desired
            # Or, if you want to ensure it runs for fixed_n_samples specifically:
            # experiments_to_run.append({
            #     "scenario": scenario,
            #     "n_samples": fixed_n_samples,
            #     "classifier_epochs_finetune": 20,
            #     "gan_epochs": 100
            # })
            pass
        else:
            experiments_to_run.append({
                "scenario": scenario,
                "n_samples": fixed_n_samples,
                "classifier_epochs_finetune": 20,
            })

    for i, exp_config in enumerate(experiments_to_run):
        exp_name = f"exp_{i+1}_{exp_config['scenario']}_N{exp_config['n_samples']}"
        exp_output_dir = os.path.join(base_output_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)
        print(f"\n--- Running Experiment: {exp_name} ---")

        # --- Step 1: Train GAN (if scenario is gan_aug) ---
        if exp_config["scenario"] == "gan_aug":
            print(f"Training GAN for {exp_config['n_samples']} samples...")
            gan_command = [
                "python", "train_gan.py",
                "--n_samples", str(exp_config["n_samples"]),
                "--epochs", str(exp_config.get("gan_epochs", 100)),
                "--output_dir", exp_output_dir, # Pass the unique output directory
                # Add other GAN parameters as needed
            ]
            try:
                subprocess.run(gan_command, check=True, cwd=os.getcwd())
                print(f"GAN training for {exp_config['n_samples']} samples completed.")
            except subprocess.CalledProcessError as e:
                print(f"Error training GAN for {exp_config['n_samples']} samples: {e}")
                continue # Skip to next experiment if GAN training fails

        # --- Step 2: Run Main Experiment ---
        main_command = [
            "python", "main_experiment.py",
            "--scenario", exp_config["scenario"],
            "--n_samples", str(exp_config["n_samples"]),
            "--batch_size", str(args.batch_size),
            "--classifier_lr", str(args.classifier_lr),
            "--classifier_epochs_mnist", str(args.classifier_epochs_mnist),
            "--classifier_epochs_finetune", str(exp_config.get("classifier_epochs_finetune", args.classifier_epochs_finetune)),
            "--gan_nz", str(args.gan_nz),
            "--gan_ngf", str(args.gan_ngf),
            "--gan_ndf", str(args.gan_ndf),
            "--gan_nc", str(args.gan_nc),
            "--gan_n_classes", str(args.gan_n_classes),
            "--n_synthetic", str(args.n_synthetic),
            "--output_dir", exp_output_dir # Pass the unique output directory
        ]
        try:
            subprocess.run(main_command, check=True, cwd=os.getcwd())
            print(f"Main experiment for {exp_name} completed.")
        except subprocess.CalledProcessError as e:
            print(f"Error running main experiment for {exp_name}: {e}")
            continue

        # --- Step 3: Organize Results ---
        # --- Step 3: Organize Results (already handled by passing output_dir) ---
        # With `main_experiment.py` and `train_gan.py` now accepting `--output_dir`,
        # all relevant artifacts (TensorBoard logs, JSON results, saved models,
        # generated images) are already saved directly into the `exp_output_dir`
        # or its subdirectories (e.g., `exp_output_dir/tensorboard_logs/`,
        # `exp_output_dir/results/`, `exp_output_dir/checkpoints/`, `exp_output_dir/gan_images/`).
        # Therefore, no explicit moving of files is needed here.
        print(f"Results for {exp_name} are organized within: {exp_output_dir}")

    print(f"\nAll experiments finished. Check {base_output_dir} for results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple domain adaptation experiments.")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for data loaders."
    )
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the classifier's Adam optimizer.",
    )
    parser.add_argument(
        "--classifier_epochs_mnist",
        type=int,
        default=5,
        help="Number of epochs to pre-train the classifier on MNIST.",
    )
    parser.add_argument(
        "--classifier_epochs_finetune",
        type=int,
        default=20,
        help="Number of epochs to fine-tune the classifier on SVHN data.",
    )
    parser.add_argument(
        "--gan_nz", type=int, default=100, help="Size of the latent vector (noise) for the GAN generator."
    )
    parser.add_argument(
        "--gan_ngf", type=int, default=64, help="Size of feature maps in the GAN generator."
    )
    parser.add_argument(
        "--gan_ndf", type=int, default=64, help="Size of feature maps in the GAN discriminator."
    )
    parser.add_argument(
        "--gan_nc", type=int, default=1, help="Number of channels in GAN input/output images."
    )
    parser.add_argument(
        "--gan_n_classes", type=int, default=10, help="Number of classes for GAN conditioning."
    )
    parser.add_argument(
        "--n_synthetic",
        type=int,
        default=5000,
        help="Number of synthetic images to generate for GAN augmentation.",
    )
    args = parser.parse_args()
    run_all_experiments(args)