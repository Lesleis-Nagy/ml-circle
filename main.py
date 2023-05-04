import os

import numpy as np

import pandas as pd

import seaborn as sns

import glob

from tqdm import tqdm

from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

np.set_printoptions(precision=3, suppress=True)


def main():

    pngs = glob.glob("output/*.png")
    imgs = {}
    for png in pngs:
        imgs[png] = np.array(Image.open(png))

    in_pngs = np.array([each for each in imgs.values()]).astype(np.float32)

    solutions = pd.read_csv("output/summary.csv")

    out_vals = solutions[["Width", "Height"]]

    # Now we build our model.
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.Conv2D(3, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=2)
    ])

    model.compile(loss='mean_squared_error', optimizer="adam")

    history = model.fit(in_pngs, out_vals, epochs=30, batch_size=200, verbose=1)


if __name__ == "__main__":
    main()
