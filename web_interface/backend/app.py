from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS # Import CORS
import os
import json
import re # Import regex for parsing experiment IDs
import torch
import torchvision.transforms as transforms
from PIL import Image
import io # For handling image bytes
from models import Classifier, Generator # Import models
from data_loader import ToTensorLong # Import ToTensorLong for target transform if needed, though not directly for inference input

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Path to the root of the DLProject, assuming this script is run from DLProject/web_interface/backend
# Adjust this path if the structure changes
DL_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPERIMENTS_DIR = os.path.join(DL_PROJECT_ROOT, 'experiments')
OUTPUTS_DIR = os.path.join(DL_PROJECT_ROOT, 'outputs')

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image preprocessing for inference
inference_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Dictionary to store loaded models to avoid reloading for every request
LOADED_CLASSIFIERS = {}

def load_classifier_model(experiment_id, scenario, n_samples):
    model_key = f"{experiment_id}_{scenario}_N{n_samples}"
    if model_key in LOADED_CLASSIFIERS:
        print(f"Using cached model for {model_key}")
        return LOADED_CLASSIFIERS[model_key]

    model = Classifier().to(DEVICE)
    
    # Determine the correct checkpoint path based on scenario
    checkpoint_filename = ""
    if scenario == "source_only":
        checkpoint_filename = "classifier_source_only.pth"
    elif scenario == "fine_tune":
        checkpoint_filename = "classifier_fine_tune.pth"
    elif scenario == "traditional_aug":
        checkpoint_filename = "classifier_traditional_aug.pth"
    elif scenario == "gan_aug":
        checkpoint_filename = "classifier_gan_aug.pth"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    model_path = os.path.join(EXPERIMENTS_DIR, experiment_id, 'exp_' + scenario + f'_N{n_samples}', 'checkpoints', checkpoint_filename)
    
    if not os.path.exists(model_path):
        # Fallback to a more general path if the specific one doesn't exist
        # This might happen if the experiment_id folder itself contains the checkpoints directly
        model_path = os.path.join(EXPERIMENTS_DIR, experiment_id, 'checkpoints', checkpoint_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Set to evaluation mode
    LOADED_CLASSIFIERS[model_key] = model
    return model


# Helper function to load experiment results from JSON files
def load_experiment_results(experiment_path):
    results = {}
    for root, _, files in os.walk(experiment_path):
        for file in files:
            if file.endswith('_results.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        scenario = data.get('scenario', 'unknown')
                        n_samples = data.get('n_samples', 'unknown')
                        # Create a unique key for each result, e.g., "source_only_N100"
                        key = f"{scenario}_N{n_samples}"
                        results[key] = data
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {file_path}")
                except Exception as e:
                    print(f"An error occurred reading {file_path}: {e}")
    return results

@app.route('/')
def index():
    return "Welcome to the Domain Adaptation Project Backend!"

@app.route('/api/experiments/summary')
def get_experiments_summary():
    all_summaries = []
    # Iterate through top-level experiment directories (e.g., 20251022-094835_N10)
    for exp_folder in os.listdir(EXPERIMENTS_DIR):
        exp_folder_path = os.path.join(EXPERIMENTS_DIR, exp_folder)
        if os.path.isdir(exp_folder_path):
            # Extract N_samples from folder name (e.g., 20251022-094835_N10 -> 10)
            match = re.search(r'_N(\d+)', exp_folder)
            n_samples = int(match.group(1)) if match else 'unknown'

            # Look for results JSON files within each scenario subdirectory
            results_data = load_experiment_results(exp_folder_path)
            
            if results_data:
                summary_entry = {
                    "experiment_id": exp_folder,
                    "n_samples": n_samples,
                    "scenarios": []
                }
                for key, data in results_data.items():
                    summary_entry["scenarios"].append({
                        "scenario_name": data.get('scenario'),
                        "final_accuracy": data.get('final_accuracy'),
                        "timestamp": data.get('timestamp'),
                        "log_dir": data.get('log_dir'),
                        "results_file": os.path.join(exp_folder, "results", f"{data.get('scenario')}_N{data.get('n_samples')}_results.json")
                    })
                all_summaries.append(summary_entry)
    return jsonify(all_summaries)

@app.route('/api/experiments/<experiment_id>/results')
def get_experiment_details(experiment_id):
    exp_path = os.path.join(EXPERIMENTS_DIR, experiment_id)
    if not os.path.isdir(exp_path):
        return jsonify({"error": "Experiment not found"}), 404
    
    results_data = load_experiment_results(exp_path)
    if not results_data:
        return jsonify({"error": "No results found for this experiment"}), 404
    
    return jsonify(results_data)

@app.route('/api/images/gan/<experiment_id>/<epoch_id>')
def get_gan_image(experiment_id, epoch_id):
    # Construct the path to the GAN image
    # Assuming GAN images are stored under experiments/<experiment_id>/gan_images/epoch_XXX.png
    image_path = os.path.join(EXPERIMENTS_DIR, experiment_id, 'gan_images', f'epoch_{epoch_id}.png')
    
    if os.path.exists(image_path):
        # Serve the image from its directory
        return send_from_directory(os.path.dirname(image_path), os.path.basename(image_path))
    else:
        return jsonify({"error": "GAN image not found"}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    # This is a placeholder for the prediction endpoint.
    # It will involve:
    # 1. Receiving an image from the frontend.
    # 2. Preprocessing the image (resize, grayscale, normalize).
    # 3. Loading the appropriate trained classifier model (e.g., based on scenario and n_samples).
    # 4. Performing inference using the loaded model.
    # 5. Returning the prediction (e.g., digit class, confidence scores).
    # This is a placeholder for the prediction endpoint.
    # It will involve:
    # 1. Receiving an image from the frontend.
    # 2. Preprocessing the image (resize, grayscale, normalize).
    # 3. Loading the appropriate trained classifier model (e.g., based on scenario and n_samples).
    # 4. Performing inference using the loaded model.
    # 5. Returning the prediction (e.g., digit class, confidence scores).
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        try:
            # Read the image file
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB') # Convert to RGB to handle various input formats

            # Apply transformations
            input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension and move to device

            # Get model parameters from request (e.g., from form data or query params)
            experiment_id = request.form.get('experiment_id')
            scenario = request.form.get('scenario')
            n_samples = request.form.get('n_samples')

            if not all([experiment_id, scenario, n_samples]):
                return jsonify({"error": "Missing experiment_id, scenario, or n_samples"}), 400
            
            n_samples = int(n_samples)

            # Load the model
            classifier_model = load_classifier_model(experiment_id, scenario, n_samples)

            # Perform inference
            with torch.no_grad():
                output = classifier_model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()

            return jsonify({
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist() # Return all probabilities
            })

        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500
    
    return jsonify({"message": "Prediction endpoint not yet implemented."}), 501


if __name__ == '__main__':
    app.run(debug=True)