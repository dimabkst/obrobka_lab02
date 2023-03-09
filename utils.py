from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


def draw_plot(xdata, ydata, xlabel: str, ylabel: str, title: str) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.grid(linestyle='--', color="black", alpha=0.4)

    plt.show()

    return fig, ax


def draw_histogram(image_histogram: list, filename: str) -> None:
    x = np.arange(0, 255 + 1, 1)
    y = np.array([el / sum(image_histogram) for el in image_histogram])
    draw_plot(x, y, "Brightness", "Frequency",
              f"Histogram of {filename}")
