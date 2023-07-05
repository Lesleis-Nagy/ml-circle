import os
import shutil

import numpy as np

import pandas as pd

import glob

from PIL import Image

import matplotlib.pyplot as plt

from typer import Typer, Argument, Option

import tensorflow as tf
import tensorflow_addons as tfa

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
        png_data = np.array(Image.open(png))
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

    print(data_sol)

    return data_obs, data_sol, width_mean, width_stdev, height_mean, height_stdev


def plot_scatter(predictions, test_sol, file_name):
    r"""
    Plot scatter containing the difference between predicted and test width/height based on input data.
    :param predictions: list of angles containing model predictions based on test data.
    :param test_sol: test solution values.
    :param file_name: the output file name.
    :return: None.
    """

    fig, axs = plt.subplots(2, 1)
    ax_width = axs[0]
    ax_height = axs[1]

    # Get the widths/heights from the test data set.
    widths_test = test_sol[:, 0]
    heights_test = test_sol[:, 1]

    # Get the widths/heights from the model predicted data set.
    widths_pred = predictions[:, 0]
    heights_pred = predictions[:, 1]

    # Create a new data frame with test and predicted widths/heights.
    widths_df = pd.DataFrame.from_dict(
        {"Test": widths_test, "Predicted": widths_pred}
    )
    heights_df = pd.DataFrame.from_dict(
        {"Test": heights_test, "Predicted": heights_pred}
    )

    # Sort by test width/height - should make the output nicer.
    widths_df_sorted = widths_df.sort_values("Test")
    heights_df_sorted = heights_df.sort_values("Test")

    # The x-axis values are simple indices.
    xs = [v for v in range(len(widths_test))]

    fig.suptitle("Ellipse width and height")

    # Plot the test and predicted widths against index.
    ax_width.plot(xs, widths_df_sorted["Test"].tolist(), label="test")
    ax_width.plot(xs, widths_df_sorted["Predicted"].tolist(), label="model")
    ax_width.set_ylabel("Width (pixels)")
    ax_width.legend(["test", "model"])

    ax_height.plot(xs, heights_df_sorted["Test"].tolist(), label="test")
    ax_height.plot(xs, heights_df_sorted["Predicted"].tolist(), label="model")
    ax_height.set_xlabel("Test index")
    ax_height.set_ylabel("Height (pixels)")
    ax_height.legend(["test", "model"])

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
                test_predictions_file: str = Argument(..., help="plot of width/height predictions."),
                loss_file: str = Argument(..., help="plot of the loss function."),
                model_weights_dir: str = Option(None, help="directory containing model weights.")):

    r"""
    Train a model to recognise ellipse widths and heights.
    """

    train_obs, train_sol, train_mean_width, train_stdev_width, train_mean_height, train_stdev_height = load_data(train_dir)
    test_obs, test_sol, test_mean_width, test_stdev_width, test_mean_height, test_stdev_height = load_data(test_dir)

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
        tf.keras.layers.Dense(units=256, activation='elu'),
        tf.keras.layers.Dense(units=128, activation='elu'),
        tf.keras.layers.Dense(units=64, activation='elu'),
        tf.keras.layers.Dense(units=2)
    ])

    # Build/compile the ANN.

    def lr_callback_fn(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr*tf.math.exp(-0.8)

    #lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_callback_fn)
    lr_callback = tfa.optimizers.CyclicalLearningRate(
        initial_learning_rate=3e-7,
        maximal_learning_rate=3e-5,
        step_size=4000,
        scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        scale_mode='cycle'
    )

    # Loss function.
    loss = tf.keras.losses.mean_squared_error

    # Optimizer to use.
    # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    optimizer = tf.keras.optimizers.legacy.Adam()

    # Metrics (accuracy is only really for classification).
    metrics = [tf.keras.metrics.Accuracy()]

    # Compile the model.
    model.compile(loss=loss, optimizer=optimizer)

    # Train the ANN.
    # history = model.fit(
    #     train_obs, train_sol, epochs=50, verbose=1, validation_split=0.2, callbacks=[lr_callback]
    # )

    history = model.fit(
        train_obs, train_sol, epochs=300, batch_size=50, verbose=1, validation_split=0.2
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
