import os
import pandas as pd
from uuid import uuid4
from PIL import Image, ImageDraw
import random


def ellipse_png(output, size=300, pad=5):
    r"""
    Generate a pdf containing an ellipsoid.
    :param output: the output PNG file.
    :param size: the (square) size of the image.
    :param pad: the padding around the image.
    :return: None
    """

    ewidth = random.randint(5, size - pad)
    eheight = random.randint(5, size - pad)

    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [(size - ewidth)/2, (size - eheight)/2, (size + ewidth)/2, (size + eheight)/2], outline="white")
    img.save(output)

    return ewidth, eheight


if __name__ == "__main__":

    OUT_DIR = "/Users/lnagy2/Projects/ml-circle/output"

    n = 10000

    summary = {
        "PNG File": [],
        "Width": [],
        "Height": []
    }

    for i in range(n):
        filename = f"{i:06d}.png"
        print(f"Generate file: {filename}")
        ewidth, eheight = ellipse_png(os.path.join(OUT_DIR, filename))
        summary["PNG File"].append(filename)
        summary["Width"].append(ewidth)
        summary["Height"].append(eheight)

    df = pd.DataFrame.from_dict(summary)

    df.to_csv(os.path.join(OUT_DIR, "summary.csv"))
