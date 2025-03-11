from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

_logger = None

def getMyModel(outputNodes):
    # basic dense model with a single hidden layer with 256 nodes
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784, 1)),
        layers.Dense(outputNodes, activation='softmax')
    ])
    
    # LeNet-5
    model = keras.Sequential([
        layers.Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)),
        layers.AveragePooling2D(pool_size=(2,2)),
        layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu'),
        layers.AveragePooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(outputNodes, activation='softmax')
    ])
    
    # ChatGTP recommended model
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # Second Convolutional Block
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),

        # Third Convolutional Block
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),

        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(outputNodes, activation='softmax')
    ])

    _logger.info('Compiling model...')
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # define our early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # return our model
    return model, early_stopping