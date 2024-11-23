from flask import Flask, request, jsonify
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Import your model evaluation functions
from model_evaluation import (
    evaluate_model_on_test_set,
    evaluate_model_on_test_set_w_confusion
)

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Get the data from the request (you can pass parameters such as model, test_loader, etc.)
    data = request.json
    model = data.get('model')  # Assuming model is passed in a suitable format
    test_loader = data.get('test_loader')  # You will need to adjust this for actual input format
    device = data.get('device', 'cpu')
    
    # Call the function for evaluation
    accuracy, precision, recall, f1 = evaluate_model_on_test_set(model, test_loader, device)
    
    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

@app.route('/evaluate_with_confusion', methods=['POST'])
def evaluate_with_confusion():
    data = request.json
    model = data.get('model')
    test_loader = data.get('test_loader')
    device = data.get('device', 'cpu')
    
    accuracy, precision, recall, f1, all_labels, all_preds = evaluate_model_on_test_set_w_confusion(model, test_loader, device)
    
    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'labels': all_labels,
        'predictions': all_preds
    })

if __name__ == '__main__':
    app.run(debug=True)
