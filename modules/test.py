import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, random_split

def evaluate_model_on_test_set(model, test_loader, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    return accuracy, precision, recall, f1, all_labels, all_preds


import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def evaluate_model_on_test_set_w_confusion(model, test_loader, device):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

    # Visualize confusion matrix
    visualize_confusion_matrix(tp, fp, tn, fn)

    return accuracy, precision, recall, f1, all_labels, all_preds


def visualize_confusion_matrix(tp, fp, tn, fn):
    # Data for visualization
    data = [[tn, fp],  # True Negatives (TN), False Positives (FP)
            [fn, tp]]  # False Negatives (FN), True Positives (TP)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap="coolwarm", interpolation='nearest')

    # Labels for the grid
    categories = ['Predicted: No', 'Predicted: Yes']
    ax.set_xticks([0, 1], labels=categories)
    ax.set_yticks([0, 1], labels=['Actual: No', 'Actual: Yes'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{data[i][j]}', ha='center', va='center', color='black', fontsize=12)

    # Title and colorbar
    plt.title("Confusion Matrix Visualization")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

