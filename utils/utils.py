from sklearn.utils import class_weight
import numpy as np
import cv2
import tensorflow as tf

def calculate_class_weights(Y_train):
    # Compute balanced class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    # Return class weights as a dictionary with class indices as keys
    return dict(enumerate(class_weights))

def convert_labels_to_indices(training_labels, test_labels):
    # Convert string labels to numerical indices for training set
    unique_labels, training_labels_indices = np.unique(training_labels, return_inverse=True)
    training_labels = training_labels_indices

    # Convert string labels to numerical indices for test set
    unique_labels, test_labels_indices = np.unique(test_labels, return_inverse=True)
    test_labels = test_labels_indices

    return training_labels, test_labels

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
    - heatmap: The Grad-CAM heatmap for the given image array.
    """

    # Ensure model.inputs is a list of KerasTensors
    if not isinstance(model.inputs, list):
        model_inputs = [model.inputs]
    else:
        model_inputs = model.inputs

    grad_model = tf.keras.models.Model(
        model_inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Avoid division by zero
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) + epsilon

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

    # Apply ReLU to focus only on positive influences
    heatmap = tf.nn.relu(heatmap)

    # Normalize
    heatmap /= tf.reduce_max(heatmap)

    # Ensure values are not too small
    heatmap = tf.maximum(heatmap, 0.1)

    return heatmap.numpy()

def find_heatmap_boundaries(heatmap):
    """
    Find the boundaries of the heatmap's activation area.

    Args:
    - heatmap: The heatmap image.

    Returns:
    - top_left: Tuple containing the coordinates of the top-left corner of the bounding rectangle.
    - bottom_right: Tuple containing the coordinates of the bottom-right corner of the bounding rectangle.
    """
    # Threshold the heatmap to obtain a binary image
    _, thresholded_heatmap = cv2.threshold(heatmap, 0.5, 1, cv2.THRESH_BINARY)

    # Convert the heatmap to uint8 for contour detection
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(heatmap_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(max_contour)

    top_left = (x, y)
    bottom_right = (x + w, y + h)
    return top_left, bottom_right
