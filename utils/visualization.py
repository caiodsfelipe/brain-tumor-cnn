import numpy as np
import matplotlib.pyplot as plt
import random

def print_class_statistics(labels, class_names, dataset_name):
    class_counts = np.bincount(labels)
    total_samples = np.sum(class_counts)
    percentage = lambda count: 100 * count / total_samples
    print(f"{dataset_name}:\n    Total: {total_samples}")
    for i, name in enumerate(class_names):
        print(f"    {name}: {class_counts[i]} ({percentage(class_counts[i]):.2f}% of total)")

def plot_random_image_examples(images, labels, class_names, num_examples=4):
    plt.figure(figsize=(20, 20))
    indices = random.sample(range(len(images)), num_examples)
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
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

    Returns:
    - None
    """
    # Generate colors and labels based on the number of folds
    colors = plt.cm.rainbow(np.linspace(0, 1, n_folds))
    labels = [f'Fold {i+1}' for i in range(n_folds)]
    
    plt.figure(figsize=(10, 7))
    plt.title(f'Train {metric_name} vs Val {metric_name}')
    for i, color in enumerate(colors):
        plt.plot(model_history[i].history[metric_name.lower()], label=f'Train {metric_name} {labels[i]}', color=color)
        plt.plot(model_history[i].history[f'val_{metric_name.lower()}'], label=f'Val {metric_name} {labels[i]}', color=color, linestyle=linestyle)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
