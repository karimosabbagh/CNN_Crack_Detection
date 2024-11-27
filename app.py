from flask import Flask, request, jsonify
from modules.prediction import load_models, predict_crack
from flask_cors import CORS  # Import CORS



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load models when the server starts
crack_models, res_net_model = load_models()

@app.route('/load_models', methods=['GET'])
def load_models_endpoint():
    """
    Endpoint to check if models are loaded.
    """
    if crack_models and res_net_model:
        return jsonify({'message': 'Models loaded successfully!'}), 200
    else:
        return jsonify({'message': 'Failed to load models.'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict cracks in an image.
    Requires JSON input: {"image_path": "<path_to_image>"}
    """
    data = request.json
    image_path = data.get('image_path')

    if not image_path:
        return jsonify({'error': 'Image path is required.'}), 400

    try:
        # Use the loaded models for prediction
        device = 'cpu'  # Change to 'cuda' if you have a GPU and want to use it
        predictions = predict_crack("Frontend/Crack_Detection"+image_path, crack_models, res_net_model, device)
        return jsonify({'predictions': predictions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
