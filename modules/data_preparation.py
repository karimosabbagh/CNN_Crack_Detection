import os
import zipfile
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
import kaggle
import numpy as np


# def authenticate_kaggle(kaggle_json_path):
#     """
#     Authenticate with Kaggle API using a kaggle.json file.
#     Args:
#         kaggle_json_path (str): Path to the kaggle.json file.
#     """
#     # kaggle_dir = os.path.expanduser("~/.kaggle")
#     kaggle_dir = "~/.kaggle"
#     os.makedirs(kaggle_dir, exist_ok=True)
#     kaggle_json_dest = os.path.join(kaggle_dir, "kaggle.json")
#     if not os.path.exists(kaggle_json_dest):
#         os.rename(kaggle_json_path, kaggle_json_dest)
#     os.chmod(kaggle_json_dest, 0o600)

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])

def authenticate_kaggle():
    kaggle_json_path = ".kaggle/kaggle.json"
    os.chmod(kaggle_json_path, 0o600)

def download_dataset(dataset_name, dataset_path):
    """
    Download a dataset from Kaggle.
    Args:
        dataset_name (str): Kaggle dataset identifier (e.g., 'arnavr10880/concrete-crack-images-for-classification').
        dataset_zip_path (str): Path where the dataset ZIP will be saved.
    """
    # Ensure the destination directory exists
    os.makedirs(dataset_path, exist_ok=True)
    # kaggle.api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)
    #if not os.path.exists(os.join(dataset_path,datas ):
    kaggle.api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)


def process_dataset_testing(data_dir, subset_fraction=1):
    """
    Process the dataset: apply transformations 
    Args:
        data_dir (str): Path to the directory where the dataset was extracted.
        subset_fraction (float): Fraction of the dataset to use for quick testing (default is 1, meaning all data).
    """
    # Load dataset with transformations
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Get class names
    class_names = dataset.classes
    return dataloader, class_names


def process_dataset_training(data_dir, subset_fraction=1):
    """
    Process the dataset: apply transformations and split into train, val, and test sets.
    Args:
        data_dir (str): Path to the directory where the dataset was extracted.
        subset_fraction (float): Fraction of the dataset to use for quick testing (default is 1, meaning all data).
    Returns:
        dict: Dataloaders for 'train', 'val', and 'test' sets.
        dict: Dataset sizes for 'train', 'val', and 'test' sets.
        list: Class names in the dataset.
        torch.device: Device to use ('cuda' or 'cpu').
    """


    # Load dataset with transformations
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    # Use a subset of the dataset if needed
    subset_size = int(len(dataset) * subset_fraction)
    _, subset_dataset = random_split(dataset, [len(dataset) - subset_size, subset_size])

    # Split the subset into train, val, and test sets
    train_size = int(0.7 * len(subset_dataset))   # 70% for training
    val_size = int(0.15 * len(subset_dataset))    # 15% for validation
    test_size = len(subset_dataset) - train_size - val_size  # Remaining 15% for testing

    train_dataset, val_dataset, test_dataset = random_split(subset_dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }

    # Get class names
    class_names = dataset.classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return dataloaders, dataset_sizes, class_names, device


if __name__ == "__main__":
    # Example usage
    kaggle_json_path = "./kaggle.json"  # Path to kaggle.json
    dataset_name = "arnavr10880/concrete-crack-images-for-classification"  # Kaggle dataset name
    dataset_zip_path = "./concrete-crack-images-for-classification.zip"  # Path to dataset ZIP
    extracted_data_dir = "./dataset"  # Path to extract dataset

    authenticate_kaggle(kaggle_json_path)
    download_dataset(dataset_name, dataset_zip_path)
    

    dataloaders, dataset_sizes, class_names, device = process_dataset(extracted_data_dir)

    print(f"Classes: {class_names}")
    print(f"Device: {device}")
    print(f"Train size: {dataset_sizes['train']}, Val size: {dataset_sizes['val']}, Test size: {dataset_sizes['test']}")

# def prepare_data():
#     # Placeholder function for data preparation
#     print("Data preparation logic goes here.")
#     return None
