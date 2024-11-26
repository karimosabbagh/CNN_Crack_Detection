import torch
from torchvision import transforms
from PIL import Image
from model import load_RESNET_model, load_model


def load_models():
    crack_model_paths = ['CNN_models/D1.pt', 'CNN_models/D2.pt', 'CNN_models/D12_simul.pt']
    crack_models = []
    for model in crack_model_paths:
        model = load_model(model)
        crack_models.append(model)
    res_net_model_path = 'CNN_models/RESNET.pt'
    res_net_model = load_RESNET_model(res_net_model_path)
    # res_net_model = None
    return crack_models , res_net_model
        

def predict_crack(image_path, crack_models, res_net_model = None, device = 'cpu'):
    """
    Predict whether an image contains a crack using multiple models.

    Args:
        image_path (str): Path to the input image.
        crack_models (list): List of loaded crack models.
        res_net_model : pretrained resnet model
        device (torch.device): Device to perform inference ('cpu' or 'cuda').

    Returns:
        dict: Predictions from each model.
    """
    # Define the image transformation pipeline
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
]   )

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    predictions = {}

    # Perform inference using each model
    for idx, model in enumerate(crack_models):
        model_name = f"Model_{idx+1}"
        model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(input_tensor)
            _, prediction = torch.max(outputs, 1)

        # Map numeric prediction to a label
        label = "Positive (Crack)" if prediction.item() == 1 else "Negative (No Crack)"
        predictions[model_name] = label

    if res_net_model != None:
        model_name = 'Trained_RESNET'
        res_net_model.to(device)
        res_net_model.eval()
        print('worked')
        with torch.no_grad():
                outputs = model(input_tensor)
                _, prediction = torch.max(outputs, 1)

        # Map numeric prediction to a label
        label = "Positive (Crack)" if prediction.item() == 1 else "Negative (No Crack)"
        predictions[model_name] = label

    return predictions



if __name__ == "__main__":
    # Load all models
    crack_models, res_net_model = load_models()

    # Predict crack for an image
    image_path = "data/D1/Negative/00001.jpg"  # Replace with your image path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = predict_crack(image_path, crack_models, res_net_model)

    # Print results

    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction}")
