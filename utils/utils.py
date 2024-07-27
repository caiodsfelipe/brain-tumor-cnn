from sklearn.utils import class_weight
import numpy as np
import cv2
import tensorflow as tf

def calculate_class_weights(y_train):
    """
    Compute balanced class weights for the training data.

    Args:
    - y_train: Training labels.

    Returns:
    - dict: Class weights as a dictionary with class indices as keys.
    """
    unique_classes = np.unique(y_train)
    class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
    return dict(enumerate(class_weights))

def convert_labels_to_indices(training_labels, test_labels):
    """
    Convert string labels to numerical indices for both training and test sets.

    Args:
    - training_labels: Labels for the training set.
    - test_labels: Labels for the test set.

    Returns:
    - tuple: Converted training labels and test labels.
    """
    _, training_indices = np.unique(training_labels, return_inverse=True)
    _, test_indices = np.unique(test_labels, return_inverse=True)
    return training_indices, test_indices

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, epsilon=1e-5):
    """
    Generate a Grad-CAM heatmap for a given image array and model.

    Args:
    - img_array: Input image array.
    - model: The model used for prediction.
    - last_conv_layer_name: Name of the last convolutional layer in the model.
    - pred_index: Index of the predicted class (default is None, which uses the predicted class with highest probability).
    - epsilon: Small value to avoid division by zero (default is 1e-5).

    Returns:
    - numpy.ndarray: The Grad-CAM heatmap for the given image array.
    """
    model_inputs = [model.inputs] if not isinstance(model.inputs, list) else model.inputs
    grad_model = tf.keras.models.Model(
        model_inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0]) if pred_index is None else pred_index
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) + epsilon

    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = tf.maximum(heatmap, 0.1)

    return heatmap.numpy()

def find_heatmap_boundaries(heatmap):
    """
    Find the boundaries of the heatmap's activation area.

    Args:
    - heatmap: The heatmap image.

    Returns:
    - tuple: Contains two tuples for the top-left and bottom-right corners of the bounding rectangle.
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    contours, _ = cv2.findContours(heatmap_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return (x, y), (x + w, y + h)
