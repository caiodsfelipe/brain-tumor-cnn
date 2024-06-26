from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras import regularizers
from setup.parameters import image_size

def get_model(dropout_rate1=0.20, dropout_rate2=0.25, dropout_rate3=0.30):
    inputs = Input(shape=(image_size, image_size, 3))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout_rate1)(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout_rate2)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout_rate3)(x)

    x = Flatten()(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    outputs = Dense(4, activation='softmax', name='last_layer')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
