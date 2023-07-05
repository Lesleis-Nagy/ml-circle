import os
import sys
import shutil
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image, ImageDraw
import random

from typer import Typer, Option, Argument

from rich.progress import track

app = Typer()


@app.command()
def ellipses(out_dir: str = Argument(..., help="directory that will contain ellipsoid images."),
             num_images: int = Argument(..., help="number of images to generate."),
             img_size: int = Option(300, help="square image size."),
             img_size_pad: int = Option(5, help="padding to use when generating ellipsoids."),
             line_thickness: int = Option(4, help="line thickness in pixels."),
             n_speckles: int = Option(50, help="the number of noise speckles to add."),
             speckle_size: int = Option(3, help="the size of a noise speckle in pixels."),
             summary_file_name: str = Option("summary.csv", help="name of the generated summary file.")):

    r"""
    Generate ellipsoid test images.
    """

    if os.path.isdir(out_dir):
        # If out_dir exists, delete it.
        response = input(f"Directory '{out_dir}' already exists, do you want to remove it? (y/N): ")
        if response.lower() == "y":
            shutil.rmtree(out_dir)
        else:
            sys.exit(0)
    # Create output file.
    os.mkdir(out_dir)

    # Summary information.
    summary = {
        "File": [],
        "Width": [],
        "Height": []
    }

    dp = len(str(num_images))

    for i in track(range(num_images), description="Generating..."):
        file_name = f"{i + 1:0{dp}d}.png"

        img = Image.new("RGB", (img_size, img_size))
        draw = ImageDraw.Draw(img)

        for j in range(n_speckles):
            speckle_x = random.randint(5, img_size - img_size_pad)
            speckle_y = random.randint(5, img_size - img_size_pad)
            draw.rectangle(
                ((speckle_x - speckle_size), (speckle_y - speckle_size),
                 (speckle_x + speckle_size), (speckle_y + speckle_size)),
                fill="grey")

        ewidth = random.randint(5, img_size - img_size_pad)
        eheight = random.randint(5, img_size - img_size_pad)

        draw.ellipse(
            [(img_size - ewidth) / 2, (img_size - eheight) / 2, (img_size + ewidth) / 2, (img_size + eheight) / 2],
            outline="white", width=line_thickness)
        img.save(os.path.join(out_dir, file_name))

        summary["File"].append(file_name)
        summary["Width"].append(ewidth)
        summary["Height"].append(eheight)

    df = pd.DataFrame.from_dict(summary)

    df.to_csv(os.path.join(out_dir, summary_file_name))

    print("Done!")


@app.command()
def lines(out_dir: str = Argument(..., help="directory that will contain line images."),
          num_images: int = Argument(..., help="number of images to generate."),
          img_size: int = Option(300, help="square image size."),
          img_size_pad: int = Option(10, help="padding to use when generating ellipsoids."),
          line_thickness: int = Option(10, help="line thickness in pixels."),
          n_speckles: int = Option(50, help="the number of noise speckles to add."),
          speckle_size: int = Option(3, help="the size of a noise speckle in pixels."),
          summary_file_name: str = Option("summary.csv", help="name of the generated summary file.")):

    r"""
    Generate line test images.
    """

    if os.path.isdir(out_dir):
        # If out_dir exists, delete it.
        response = input(f"Directory '{out_dir}' already exists, do you want to remove it? (y/N): ")
        if response.lower() == "y":
            shutil.rmtree(out_dir)
        else:
            sys.exit(0)
    # Create output file.
    os.mkdir(out_dir)

    # Summary information.
    summary = {
        "File": [],
        "Angle": []
    }

    dp = len(str(num_images))

    for i in track(range(num_images), description="Generating..."):
        file_name = f"{i+1:0{dp}d}.png"

        img = Image.new("RGB", (img_size, img_size), color="white")
        draw = ImageDraw.Draw(img)

        angle = random.uniform(0.0, math.pi)

        sx = -float(img_size - img_size_pad)/2.0 * math.cos(angle) + img_size/2.0
        sy = -float(img_size - img_size_pad)/2.0 * math.sin(angle) + img_size/2.0

        ex = float(img_size - img_size_pad)/2.0 * math.cos(angle) + img_size/2.0
        ey = float(img_size - img_size_pad)/2.0 * math.sin(angle) + img_size/2.0

        for j in range(n_speckles):
            speckle_x = random.randint(5, img_size - img_size_pad)
            speckle_y = random.randint(5, img_size - img_size_pad)
            draw.rectangle(
                ((speckle_x - speckle_size), (speckle_y - speckle_size),
                 (speckle_x + speckle_size), (speckle_y + speckle_size)),
                fill="grey")

        draw.line((sx, sy, ex, ey), fill="black", width=line_thickness)

        img.save(os.path.join(out_dir, file_name))

        summary["File"].append(file_name)
        summary["Angle"].append(angle)

    df = pd.DataFrame.from_dict(summary)
    df.to_csv(os.path.join(out_dir, summary_file_name))


def main():
    r"""
    Main function.
    :return: None
    """
    app()


if __name__ == "__main__":
    main()
