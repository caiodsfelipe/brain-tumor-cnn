import tensorflow as tf
from models.model import *
from setup.parameters import epochs, batch_size
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def get_training_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=18)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='../weights/weights.keras',
        verbose=1,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    return [early_stopping, reduce_lr, model_checkpoint]

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
