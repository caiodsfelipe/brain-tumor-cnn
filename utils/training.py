import tensorflow as tf
from models.model import get_model
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
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=18
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.2,
        patience=5
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='../weights/weights.keras',
        verbose=1,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    
    return [early_stopping, reduce_lr, model_checkpoint]

def fit_and_evaluate(training_data, validation_data, class_weights, callbacks):
    """
    Trains a model using the provided training data and evaluates it on the validation data.

    Args:
    - training_data: Tuple of (training_images, training_labels)
    - validation_data: Tuple of (validation_images, validation_labels)
    - class_weights: Weights to apply to the loss function during training, useful for imbalanced datasets.
    - callbacks: List of callbacks to use during training.

    Returns:
    - results: The training results, such as loss and accuracy, for each epoch.
    """
    model = get_model()

    training_images, training_labels = training_data
    validation_images, validation_labels = validation_data

    results = model.fit(
        training_images, 
        training_labels, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=(validation_images, validation_labels),
        class_weight=class_weights,
        callbacks=callbacks, 
        verbose=1
    )

    print("\nValidation Score:", model.evaluate(validation_images, validation_labels))
    return results

def run_cross_validation(n_folds, training_data, class_weights, callbacks):
    """
    Performs k-fold cross-validation on the training data.

    Args:
    - n_folds: Number of folds for cross-validation.
    - training_data: Tuple of (training_images, training_labels)
    - class_weights: Weights to apply to the loss function during training.
    - callbacks: List of callbacks to use during training.

    Returns:
    - model_history: List of training histories for each fold.
    """
    training_images, training_labels = training_data
    model_history = []

    for i in range(n_folds):
        print(f"Training on Fold: {i + 1}")

        split_data = train_test_split(training_images, training_labels, test_size=0.2)
        split_training_images, split_validation_images, split_training_labels, split_validation_labels = split_data

        split_training_data = shuffle(split_training_images, split_training_labels)
        split_validation_data = (split_validation_images, split_validation_labels)

        fold_history = fit_and_evaluate(split_training_data, split_validation_data, class_weights, callbacks)
        model_history.append(fold_history)
        print("=======" * 12, end="\n\n\n")

    return model_history
