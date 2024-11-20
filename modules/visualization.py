import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



# Helper function to display images
def show_grayscale_images(dataset, num_images=6):
    # Create a DataLoader for the dataset
    data_loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

    # Get a batch of images
    images, labels = next(iter(data_loader))

    # Plot the images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i].squeeze(0), cmap="gray")  # Remove channel dim and use grayscale colormap
        ax.set_title(f"Label: {labels[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_results(results):
    # Placeholder function for visualization
    print("Visualization logic goes here.")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_enhanced_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot an enhanced confusion matrix with actual counts and percentages.

    Args:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.
        class_names (list): List of class names (e.g., ['Positive', 'Negative']).
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Calculate percentages

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot confusion matrix with percentages
    im = ax.imshow(cm, interpolation='nearest', cmap='coolwarm')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Counts", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[f'Predicted: {cls}' for cls in class_names],
           yticklabels=[f'Actual: {cls}' for cls in class_names],
           title='Enhanced Confusion Matrix',
           ylabel='Actual Value',
           xlabel='Predicted Value')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.0f'  # Format for counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count_text = f"{cm[i, j]:.0f}"  # Actual counts
            percent_text = f"\n({cm_percentage[i, j]:.1f}%)"  # Percentages
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, count_text + percent_text,
                    ha="center", va="center", color=text_color)

    fig.tight_layout()
    plt.show()

