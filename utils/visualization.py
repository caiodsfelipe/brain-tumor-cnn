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
    # Count occurrences of each class label
    class_counts = np.bincount(labels)
    # Calculate total number of samples
    total_samples = np.sum(class_counts)
    # Define a lambda function to calculate percentage
    percentage = lambda count: 100 * count / total_samples
    # Print dataset name and total number of samples
    print(f"{dataset_name}:\n    Total: {total_samples}")
    # Print count and percentage for each class
    for i, name in enumerate(class_names):
        print(f"    {name}: {class_counts[i]} ({percentage(class_counts[i]):.2f}% of total)")

def plot_random_image_examples(images, labels, class_names, num_examples=4):
    """
    Plots a grid of randomly selected image examples from the dataset, with their corresponding class labels.
    
    Args:
        images (array-like): Array of images to be plotted.
        labels (array-like): Array of class labels corresponding to the images.
        class_names (list of str): List of class names, indexed according to class labels.
        num_examples (int): Number of random examples to plot (default is 4).
    """
    # Set up the figure with a specific size
    plt.figure(figsize=(20, 20))
    # Randomly sample indices of images to plot
    indices = random.sample(range(len(images)), num_examples)
    # Plot each selected image
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)  # Create a subplot for each image
        plt.imshow(images[idx])   # Display the image
        plt.xlabel(class_names[labels[idx]], fontsize=24)  # Set the class label as xlabel
    plt.show()  # Display the plot


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
