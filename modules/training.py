import time
from tempfile import TemporaryDirectory
import os
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from copy import deepcopy
import torch.optim as optim
from torch.optim import lr_scheduler
import shutil

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=11, params_file='best_model_params.pt'):
    since = time.time()
    # Directory to save model parameters
    model_params_dir = 'CNN_models'
    os.makedirs(model_params_dir, exist_ok=True)  # Ensure the directory exists

    # Path to save the best model parameters permanently
    permanent_best_model_path = os.path.join(model_params_dir, params_file)
    best_acc = 0.0

    # Metrics
    train_losses = []
    val_losses = []
    val_precision = []
    val_recall = []
    val_f1 = []
    val_accuracies = []

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        temp_best_model_path = os.path.join(tempdir, params_file)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                all_preds = []
                all_labels = []

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if phase == 'val':
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_losses.append(epoch_loss)
                else:
                    val_losses.append(epoch_loss)
                    val_accuracies.append(epoch_acc.item())

                    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
                    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
                    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

                    val_precision.append(precision)
                    val_recall.append(recall)
                    val_f1.append(f1)

                    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

                # Save the best model temporarily and permanently
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), temp_best_model_path)
                    shutil.copy(temp_best_model_path, permanent_best_model_path)
                    print(f'Saving best model weights to {permanent_best_model_path}...')

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(torch.load(permanent_best_model_path))
    return model, best_acc, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1
    }


# def train_model(model, criterion, optimizer, scheduler,  dataloaders, dataset_sizes, device, num_epochs=11, params_file='best_model_params.pt'):
#     since = time.time()
#     # Directory to save model parameters
#     model_params_dir = 'CNN_models'

#     best_model_params_path = os.path.join(model_params_dir, params_file)
#     # drive_save_path = 'models/'

#     # Create a temporary directory to save training checkpoints
#     with TemporaryDirectory() as tempdir:
#         best_model_params_path = os.path.join(tempdir, params_file)

#         torch.save(model.state_dict(), best_model_params_path)
#         best_acc = 0.0

#         # metrics
#         train_losses = []
#         val_losses = []
#         val_precision = []
#         val_recall = []
#         val_f1 = []
#         val_accuracies = []

#         for epoch in range(num_epochs):
#             print(f'Epoch {epoch}/{num_epochs - 1}')
#             print('-' * 10)

#             # Each epoch has a training and validation phase
#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     model.train()  # Set model to training mode
#                 else:
#                     model.eval()   # Set model to evaluate mode

#                 running_loss = 0.0
#                 running_corrects = 0
#                 all_preds = []
#                 all_labels = []

#                 # Iterate over data.
#                 for inputs, labels in dataloaders[phase]:
#                     inputs = inputs.to(device)
#                     labels = labels.to(device)

#                     # zero the parameter gradients
#                     optimizer.zero_grad()

#                     # forward
#                     # track history if only in train
#                     with torch.set_grad_enabled(phase == 'train'):
#                         outputs = model(inputs)
#                         _, preds = torch.max(outputs, 1)
#                         loss = criterion(outputs, labels)

#                         # backward + optimize only if in training phase
#                         if phase == 'train':
#                             loss.backward()
#                             optimizer.step()

#                     # statistics
#                     running_loss += loss.item() * inputs.size(0)
#                     running_corrects += torch.sum(preds == labels.data)

#                     if phase == 'val':
#                       all_preds.extend(preds.cpu().numpy())
#                       all_labels.extend(labels.cpu().numpy())

#                 if phase == 'train':
#                     scheduler.step()

#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 epoch_acc = running_corrects.double() / dataset_sizes[phase]

#                 print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#                 if phase == 'train':
#                   train_losses.append(epoch_loss)
#                 else:
#                   val_losses.append(epoch_loss)
#                   val_accuracies.append(epoch_acc.item())

#                   precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
#                   recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
#                   f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

#                   val_precision.append(precision)
#                   val_recall.append(recall)
#                   val_f1.append(f1)

#                   print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

#                 # deep copy the model
#                 if phase == 'val' and epoch_acc > best_acc:
#                     best_acc = epoch_acc
#                     torch.save(model.state_dict(), best_model_params_path)
#                     print('Saving best model weights...')
#                     # shutil.copy(best_model_params_path, drive_save_path)

#             # print()

#         time_elapsed = time.time() - since
#         print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#         print(f'Best val Acc: {best_acc:4f}')

#         # load best model weights
#         model.load_state_dict(torch.load(best_model_params_path, weights_only=True))

    return model, best_acc, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1
    }

def k_fold_cross_validation(model, dataset, criterion, optimizer_fn, scheduler_fn, k=5, num_epochs=11, batch_size=32, device='cpu', params_file='best_model_params.pt'):   

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'train_losses': [],
        'val_losses': [],
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold+1}/{k}')
        print('-' * 20)

        # Create training and validation subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create dataloaders for training and validation
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        dataset_sizes = {
            'train': len(train_subset),
            'val': len(val_subset)
        }

        # Initialize a new model instance for this fold
        model = deepcopy(model).to(device)

        # Define optimizer and scheduler for this fold
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer)

        # Train the model for this fold
        trained_model, best_acc, metrics = train_model(
            model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs, params_file
        )

        # Collect metrics
        fold_results['accuracy'].append(np.max(metrics['val_accuracies']))
        fold_results['precision'].append(np.mean(metrics['precision']))
        fold_results['recall'].append(np.mean(metrics['recall']))
        fold_results['f1'].append(np.mean(metrics['f1']))
        fold_results['train_losses'].append(metrics['train_losses'])
        fold_results['val_losses'].append(metrics['val_losses'])

        print(f'Fold {fold+1} Best Accuracy: {fold_results["accuracy"][-1]:.4f}')
        print(f'Precision: {fold_results["precision"][-1]:.4f}, Recall: {fold_results["recall"][-1]:.4f}, F1-Score: {fold_results["f1"][-1]:.4f}')
        print()

    # Average metrics across folds
    print(f'K-Fold Cross Validation Results:')
    print(f'Mean Accuracy: {np.mean(fold_results["accuracy"]):.4f}')
    print(f'Mean Precision: {np.mean(fold_results["precision"]):.4f}')
    print(f'Mean Recall: {np.mean(fold_results["recall"]):.4f}')
    print(f'Mean F1-Score: {np.mean(fold_results["f1"]):.4f}')

    # Return metrics
    return fold_results, model





def train_model_k_fold(model, dataloaders, criterion, k = 5, num_epochs=11, batch_size = 32, device='cpu', params_file='best_model_params.pt'):
    train_dataloader, val_dataloader, test_dataloaders = dataloaders.values()
    train_dataset = train_dataloader.dataset
    val_dataset = val_dataloader.dataset
    train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    results, model = k_fold_cross_validation(
        model= model,
        dataset= train_val_dataset,
        criterion=criterion,
        optimizer_fn= lambda params: optim.SGD(params, lr=0.001, momentum=0.9),
        scheduler_fn= lambda optimizer: lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
        k=k,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        params_file=params_file
    )
    return model, results

