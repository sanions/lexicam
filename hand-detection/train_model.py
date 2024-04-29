import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models

from load_data import load_images, load_labels

# Define the CNN model
def create_model(input_shape=(576, 720, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(4)  # Output layer for bounding box coordinates (x, y, w, h)
    ])
    return model


def train(): 
    # create an instance of the model
    model = create_model()

    # compile the model
    model.compile(optimizer='adam',
                loss='mean_squared_error',  # Using MSE loss for bounding box regression
                metrics=['accuracy'])

    # define dataset and train the model
    datapath = os.path.join(os.path.dirname(__file__), "outputs/hands_bbox_train.csv")
    train_df = pd.read_csv(datapath, index_col=0)
    train_images = load_images(train_df)
    train_labels = load_labels(train_df)  

    datapath = os.path.join(os.path.dirname(__file__), "outputs/hands_bbox_test.csv")
    test_df = pd.read_csv(datapath, index_col=0)
    val_images = load_images(test_df)
    val_labels = load_labels(test_df)

    # Train the model
    hist = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

    print(model.summary())

    model.save("hand_model.h5")

    # Evaluate the model
    test_loss, test_acc = model.evaluate(val_images, val_labels)

    # Print test accuracy
    print('Test accuracy:', test_acc)


if __name__ == "__main__": 
    train()

