import numpy as np
import matplotlib.pyplot as plt
import random

def print_class_statistics(labels, class_names, dataset_name):
    """
    Prints the number of samples for each class in the dataset and their percentage of the total dataset.
    
    Args:
        labels (array-like): Array of class labels.
        class_names (list of str): List of class names, indexed according to class labels.
        dataset_name (str): Name of the dataset to include in the output.
    """
    class_counts = np.bincount(labels)
    total_samples = np.sum(class_counts)
    
    print(f"{dataset_name}:\n    Total: {total_samples}")
    for i, name in enumerate(class_names):
        count = class_counts[i]
        percentage = 100 * count / total_samples
        print(f"    {name}: {count} ({percentage:.2f}% of total)")

def plot_random_image_examples(images, labels, class_names, num_examples=4):
    """
    Plots a grid of randomly selected image examples from the dataset, with their corresponding class labels.
    
    Args:
        images (array-like): Array of images to be plotted.
        labels (array-like): Array of class labels corresponding to the images.
        class_names (list of str): List of class names, indexed according to class labels.
        num_examples (int): Number of random examples to plot (default is 4).
    """
    plt.figure(figsize=(20, 20))
    indices = random.sample(range(len(images)), num_examples)
    for i, idx in enumerate(indices, 1):
        plt.subplot(5, 5, i)
        plt.imshow(images[idx])
        plt.xlabel(class_names[labels[idx]], fontsize=24)
    plt.show()

def plot_metrics(model_history, metric_name, ylabel, n_folds, linestyle=''):
    """
    Plot the training and validation metrics for each fold.

    Args:
    - model_history: List of Keras History objects.
    - metric_name: Name of the metric (e.g., 'loss', 'accuracy').
    - ylabel: Label for the y-axis.
    - n_folds: Number of folds.
    - linestyle: Style of the line (default is solid).
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, n_folds))
    
    plt.figure(figsize=(10, 7))
    plt.title(f'Train {metric_name} vs Val {metric_name}')
    
    for i, (history, color) in enumerate(zip(model_history, colors), 1):
        train_metric = history.history[metric_name.lower()]
        val_metric = history.history[f'val_{metric_name.lower()}']
        
        plt.plot(train_metric, label=f'Train {metric_name} Fold {i}', color=color)
        plt.plot(val_metric, label=f'Val {metric_name} Fold {i}', color=color, linestyle=linestyle)
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
