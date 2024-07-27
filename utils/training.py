import tensorflow as tf
from models.model import *
from setup.parameters import epochs, batch_size
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def get_training_callbacks():
    """
    Creates and returns a list of TensorFlow Keras callbacks for training a model.
    
    Returns:
        list: A list containing three TensorFlow Keras callbacks:
            - EarlyStopping: Stops training when a monitored metric has stopped improving.
            - ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving.
            - ModelCheckpoint: Saves the model after every epoch with the best validation accuracy.
    """
    # Callback to stop training early if the validation loss does not improve for a certain number of epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=18          # Number of epochs with no improvement after which training will be stopped
    )
    
    # Callback to reduce learning rate when the training loss plateaus
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',      # Metric to monitor
        factor=0.2,          # Factor by which the learning rate will be reduced
        patience=5           # Number of epochs with no improvement after which learning rate will be reduced
    )
    
    # Callback to save the model with the best validation accuracy
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='../weights/weights.keras',  # Path where the model weights will be saved
        verbose=1,                            # Verbosity mode, 1 for progress messages
        monitor='val_accuracy',               # Metric to monitor for saving the model
        mode='max',                           # Mode for the metric, 'max' to save the best model with highest accuracy
        save_best_only=True                    # Only save the model if the monitored metric has improved
    )
    
    return [early_stopping, reduce_lr, model_checkpoint]  # Return the list of callbacks

def fit_and_evaluate(training_images, validation_images, training_labels, validation_labels, class_weights, callbacks):
    """
    Trains a model using the provided training data and evaluates it on the validation data.

    Args:
    - training_images: Input training images.
    - validation_images: Input validation images.
    - training_labels: Labels for the training images.
    - validation_labels: Labels for the validation images.
    - class_weights: Weights to apply to the loss function during training, useful for imbalanced datasets.

    Returns:
    - results: The training results, such as loss and accuracy, for each epoch.
    """

    model = get_model()

    results = model.fit(training_images, training_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(validation_images, validation_labels),
                        class_weight=class_weights,
                        callbacks=callbacks, verbose=1)

    # Evaluate the model on the validation data and print the validation score
    print("\nValidation Score: ", model.evaluate(validation_images, validation_labels))
    return results

def run_cross_validation(n_folds, training_images, training_labels, class_weights, callbacks):
    model_history = []

    for i in range(n_folds):
        print("Training on Fold: ", i + 1)

        # Split the training data into training and validation sets
        split_training_images, split_validation_images, split_training_labels, split_validation_labels = train_test_split(
            training_images, training_labels, test_size=0.2
        )

        # Shuffle the training data
        split_training_images, split_training_labels = shuffle(split_training_images, split_training_labels)

        # Train and evaluate the model on the current fold
        model_history.append(fit_and_evaluate(split_training_images, split_validation_images, split_training_labels, split_validation_labels, class_weights, callbacks))
        print("=======" * 12, end="\n\n\n")

    return model_history
