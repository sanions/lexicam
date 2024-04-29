import tensorflow as tf
from tensorflow.keras import layers, models

# define the model architecture
def create_model():
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        # Fully connected layers
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        # Output layer with 42 neurons (21 landmarks * 2 coordinates)
        layers.Dense(42)
    ])
    return model


def train():
    # Load  labeled data (images and corresponding landmarks)
    # X_train, y_train = ...
    # X_test, y_test = ...

    # Define the loss function
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Instantiate the model
    model = hand_pose_detection_model()

    # Compile the model
    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['mae', 'mse'])

    # Train the model
    hist = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    print(model.summary())

    # Evaluate the model
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test MAE:", test_mae)
    print("Test MSE:", test_mse)

    model.save("finger_model.h5")


if __name__ = "__main__": 
    train()