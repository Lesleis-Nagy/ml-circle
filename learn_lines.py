import sys
import os
import shutil

import numpy as np

import pandas as pd

import glob

from PIL import Image

import matplotlib.pyplot as plt

from typer import Typer, Argument, Option

import tensorflow as tf

np.set_printoptions(precision=3, suppress=True)

app = Typer()


def load_data(data_dir):
    r"""
    Loads image and summary data from data_dir.
    :param data_dir: the directory containing images (in PNG format) along with a summary.csv file.
    :return: pair consisting of data observations (the images) and data solutions (the angles).
    """
    data_png_file_names = sorted(glob.glob(f"{data_dir}/*.png"))
    data_pngs = {}
    for png in data_png_file_names:
        data_pngs[png] = np.array(Image.open(png).convert('RGB'))
    data_csv = pd.read_csv(f"{data_dir}/summary.csv")
    data_obs = np.array([each for each in data_pngs.values()]).astype(np.float32)
    data_sol = data_csv[["Angle"]].astype(np.float32)

#    data_obs = tf.image.per_image_standardization(data_obs)

    return data_obs, data_sol


def plot_scatter(predictions, test_sol, file_name):
    r"""
    Plot scatter containing the difference between predicted and test angles based on input data.
    :param predictions: list of angles containing model predictions based on test data.
    :param test_sol: test solution values.
    :param file_name: the output file name.
    :return: None.
    """

    fig, ax = plt.subplots()

    fig.suptitle("Line angles")

    # Get the angles from the test data set.
    angles_test = test_sol["Angle"].tolist()

    # Get the angles from the model predicted data set.
    angles_pred = [r[0] for r in predictions]

    # Create a new data frame with test and predicted angles.
    angles_df = pd.DataFrame.from_dict(
        {"Test": angles_test, "Predicted": angles_pred}
    )

    # Sort by test agnes - should make the output nicer.
    angles_df_sorted = angles_df.sort_values("Test")

    # The x-axis values are simple indices.
    xs = [v for v in range(len(angles_test))]

    # Plot the test and predicted widths against index.
    ax.plot(xs, angles_df_sorted["Test"].tolist(), label="test")
    ax.plot(xs, angles_df_sorted["Predicted"].tolist(), label="model")
    ax.legend(["test", "model"])

    fig.savefig(file_name)
    plt.close(fig)


def plot_loss(history, file_name):
    r"""
    Plot loss data, extracted form a tensorflow history object.
    :param history: the tensorflow history object.
    :param file_name: the file name of the output plot.
    :return: None.
    """

    fig, ax = plt.subplots()

    n = len(history.history['loss']) - 1

    x = [i+2 for i in range(n)]
    y_loss = [v for v in history.history["loss"][1:]]
    y_val_loss = [v for v in history.history["val_loss"][1:]]

    fig.suptitle("Training data")

    ax.plot(x, y_loss, label='loss function')
    ax.plot(x, y_val_loss, label='validation error')

    ax.set_xlabel('Epoch')

    ax.set_ylabel('Mean squared error')
    #ax.set_yscale("log")

    ax.legend(["loss function", "validation error"])

    fig.savefig(file_name)
    plt.close(fig)


@app.command()
def train_model(train_dir:str =Argument(..., help="directory containing training images."),
                test_dir: str = Argument(..., help="directory containing test images."),
                test_predictions_file: str = Argument(..., help="plot of angle predictions."),
                loss_file: str = Argument(..., help="plot of the loss function."),
                model_weights_dir: str = Option(None, help="directory containing model weights.")):

    r"""
    Train a model to recognise ellipse widths and heights.
    """

    train_obs, train_sol = load_data(train_dir)
    test_obs, test_sol = load_data(test_dir)

    # Configure the ANN.
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.SpatialDropout2D(0.2),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1, kernel_constraint=tf.keras.constraints.NonNeg())
    ])

    # Build/compile the ANN.

    def lr_callback_fn(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr*tf.math.exp(-0.1)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_callback_fn)

    # Loss function.
    loss = tf.keras.losses.mean_squared_error

    # Optimizer to use.
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)

    # Metrics (accuracy is only really for classification).
    metrics = [tf.keras.metrics.Accuracy()]

    # Compile the model.
    model.compile(loss=loss, optimizer=optimizer)

    # Train the ANN.
    # history = model.fit(
    #     train_obs, train_sol, epochs=30, verbose=1, validation_split=0.2, callbacks=[lr_callback]
    # )

    history = model.fit(
        train_obs, train_sol, epochs=300, verbose=1, validation_split=0.2
    )

    # Get predictions based on test objects.
    predictions = model.predict(test_obs)

    # Plot some output.
    plot_scatter(predictions, test_sol, test_predictions_file)
    plot_loss(history, loss_file)

    # Save the weights (if requested).
    if model_weights_dir is not None:
        if os.path.isdir(model_weights_dir):
            shutil.rmtree(model_weights_dir)
        model.save_weights(model_weights_dir)


def main():
    app()


if __name__ == "__main__":
    main()
