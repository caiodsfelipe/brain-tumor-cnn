from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras import regularizers
from setup.parameters import image_size

def get_model(dropout_rate1=0.20, dropout_rate2=0.25, dropout_rate3=0.30):
    """
    Constructs and compiles a Convolutional Neural Network (CNN) model with dropout layers.
    
    Args:
        dropout_rate1 (float): Dropout rate after the first convolutional layer.
        dropout_rate2 (float): Dropout rate after the second convolutional layer.
        dropout_rate3 (float): Dropout rate after the third convolutional layer.
    
    Returns:
        model: A compiled Keras Model instance.
    """
    # Input layer with shape corresponding to the image dimensions and number of channels (RGB)
    inputs = Input(shape=(image_size, image_size, 3))
    
    # First convolutional layer with 16 filters, ReLU activation, and same padding
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    # Max pooling layer to reduce spatial dimensions by a factor of 2
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Dropout layer to prevent overfitting, applying dropout with specified rate
    x = Dropout(dropout_rate1)(x)
    
    # Second convolutional layer with 32 filters, ReLU activation, and same padding
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # Max pooling layer to reduce spatial dimensions by a factor of 2
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Dropout layer to prevent overfitting, applying dropout with specified rate
    x = Dropout(dropout_rate2)(x)
    
    # Third convolutional layer with 64 filters, ReLU activation, and same padding
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # Max pooling layer to reduce spatial dimensions by a factor of 2
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Dropout layer to prevent overfitting, applying dropout with specified rate
    x = Dropout(dropout_rate3)(x)
    
    # Flatten layer to convert 3D tensor to 1D vector
    x = Flatten()(x)
    # Dense layer with 32 units, ReLU activation, and L2 regularization
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    # Output layer with 4 units (one for each class), softmax activation for multi-class classification
    outputs = Dense(4, activation='softmax', name='last_layer')(x)
    
    # Create the model with specified inputs and outputs
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
