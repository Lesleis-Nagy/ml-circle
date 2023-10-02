import os
import sys
import shutil

import numpy as np

import pandas as pd

import glob

from PIL import Image, ImageOps

import matplotlib.pyplot as plt

from typer import Typer, Argument, Option

import tensorflow as tf

np.set_printoptions(precision=3, suppress=True)

app = Typer()


def load_data(data_dir):
    r"""
    Loads image and summary data from data_dir.
    :param data_dir: the directory containing images (in PNG format) along with a summary.csv file.
    :return: pair consisting of data observations (the images) and data solutions (the ellipse width/height).
    """
    data_png_file_names = sorted(glob.glob(f"{data_dir}/*.png"))
    data_pngs = {}
    for png in data_png_file_names:
        print(f"PNG: {png}")
        png_data = np.array(ImageOps.grayscale(Image.open(png)), dtype=np.float64)
        png_data = (png_data - 128) / 128.0

        png_norm = (png_data - np.mean(png_data))/np.sqrt(np.var(png_data))
        data_pngs[png] = png_norm

    data_csv = pd.read_csv(f"{data_dir}/summary.csv")
    data_obs = np.array([each for each in data_pngs.values()]).astype(np.float32)

    data_sol = data_csv[["Width", "Height"]].astype(np.float32)

    width_mean = np.mean(data_csv["Width"])
    width_stdev = np.std(data_csv["Width"])
    height_mean = np.mean(data_csv["Height"])
    height_stdev = np.std(data_csv["Height"])

    data_sol = np.column_stack([(data_csv["Width"].to_numpy() - width_mean) / width_stdev,
                                (data_csv["Height"].to_numpy() - height_mean) / height_stdev])

    return data_obs, data_sol, width_mean, width_stdev, height_mean, height_stdev


def write_scatter(model_predictions, actual_data, width_file, height_file):
    r"""
    write scatter data containing the difference between predicted and test width/height based on input data.
    :param model_predictions: list of angles containing model predictions based on test data.
    :param actual_data: test solution values.
    :param file_name: the output file name.
    :return: None.
    """

    # Get the widths/heights from the test data set.
    width_actual = actual_data[:, 0]
    height_actual = actual_data[:, 1]

    # Get the widths/heights from the model predicted data set.
    width_pred = model_predictions[:, 0]
    height_pred = model_predictions[:, 1]

    width_df = pd.DataFrame(
        {"Actual width": width_actual,
         "Predicted width": width_pred}
    ).sort_values("Actual width")

    height_df = pd.DataFrame(
        {"Actual height": height_actual,
         "Predicted height": height_pred}
    ).sort_values("Actual height")

    width_df.to_csv(width_file)
    height_df.to_csv(height_file)


def write_loss(history, file_name):
    r"""
    Plot loss data, extracted form a tensorflow history object.
    :param history: the tensorflow history object.
    :param file_name: the file name of the output plot.
    :return: None.
    """

    n = len(history.history['loss']) - 1

    x = [i+2 for i in range(n)]
    y_loss = [v for v in history.history["loss"][1:]]
    y_val_loss = [v for v in history.history["val_loss"][1:]]

    df = pd.DataFrame({
        "Epoch": x,
        "Loss": y_loss,
        "Value loss": y_val_loss
    })

    df.to_csv(file_name)


@app.command()
def train_model(train_dir: str = Argument(..., help="directory containing training images."),
                test_dir: str = Argument(..., help="directory containing test images."),
                width_predictions_file: str = Argument(..., help="plot of width predictions."),
                height_predictions_file: str = Argument(..., help="plot of height predictions."),
                loss_file: str = Argument(..., help="plot of the loss function."),
                model_weights_dir: str = Option(None, help="directory containing model weights.")):

    r"""
    Train a model to recognise ellipse widths and heights.
    """

    train_obs, train_sol, train_mean_width, train_stdev_width, train_mean_height, train_stdev_height = load_data(train_dir)
    test_obs, test_sol, test_mean_width, test_stdev_width, test_mean_height, test_stdev_height = load_data(test_dir)

    batch_size = 50
    height = 300
    width = 300
    channels = 1

    # Configure the ANN.
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3, 3, activation='relu', input_shape=(height, width, channels)),
        tf.keras.layers.Conv2D(3, 3, activation='relu', input_shape=(height, width, channels)),
        tf.keras.layers.MaxPooling2D(2, input_shape=(height, width, channels)),
        tf.keras.layers.Conv2D(3, 3, activation='relu', input_shape=(height, width, channels)),
        tf.keras.layers.Conv2D(3, 3, activation='relu', input_shape=(height, width, channels)),
        tf.keras.layers.MaxPooling2D(2, input_shape=(height, width, channels)),
        tf.keras.layers.Conv2D(3, 3, activation='relu', input_shape=(height, width, channels)),
        tf.keras.layers.Conv2D(3, 3, activation='relu', input_shape=(height, width, channels)),
        tf.keras.layers.SpatialDropout2D(0.2, input_shape=(height, width, channels)),
        tf.keras.layers.MaxPooling2D(2, input_shape=(height, width, channels)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation='elu'),
        tf.keras.layers.Dense(units=128, activation='elu'),
        tf.keras.layers.Dense(units=64, activation='elu'),
        tf.keras.layers.Dense(units=2)
    ])

    # Loss function.
    loss = tf.keras.losses.mean_squared_error

    # Optimizer to use.
    optimizer = tf.keras.optimizers.legacy.Adam()
    #optimizer = tf.keras.optimizers.Adam()

    # Compile the model.
    model.compile(loss=loss, optimizer=optimizer)

    history = model.fit(
        train_obs, train_sol, epochs=300, batch_size=batch_size, verbose=1, validation_split=0.2
    )

    # Get predictions based on test objects.
    predictions = model.predict(test_obs)

    # Plot some output.
    write_scatter(predictions, test_sol, width_predictions_file, height_predictions_file)
    write_loss(history, loss_file)

    # Save the weights (if requested).
    if model_weights_dir is not None:
        if os.path.isdir(model_weights_dir):
            shutil.rmtree(model_weights_dir)
        model.save_weights(model_weights_dir)


def main():
    app()


if __name__ == "__main__":
    main()
