from typing import Tuple, Dict, List
from os import path
from PIL import Image
from math import sqrt
import numpy as np


def one_byte_projection(old_intensity: float, min_old_intensity: float, max_old_intensity: float) -> int:
    return 255 * (old_intensity - min_old_intensity) // (max_old_intensity - min_old_intensity)


def Roberts_operator(Gx: int, Gy: int, delta: float = 100) -> int:
    return int(sqrt(Gx ** 2 + Gy ** 2) > delta)
    # return int((abs(Gx) >= delta or abs(Gy) >= delta))  # alternative


def Sobel3_operator(a: int, b: int, c: int, d: int, e: int, g: int, h: int, i: int) -> Tuple[int, int]:
    # actually Gy, used name from lab file
    Gx = g + 2 * h + i - (a + 2 * b + c)
    Gy = c + 2 * e + i - (a + 2 * d + g)  # actually Gx

    return (Gx, Gy)


def Sobel3_operator_boundaries(image: Image.Image) -> Dict[Image.Image, Image.Image]:
    image_pixels = [[image.getpixel((i, j)) for j in range(
        image.width)] for i in range(image.height)]  # save it, so can use more efficiently in next loop
    gradients = []
    sob3_image_pixels = []
    sob3_roberts_image_pixels = []

    # finding gradients + sobel3 with roberts operator image
    for y in range(1, image.height - 1):
        gradients.append([])
        sob3_roberts_image_pixels.append([])
        for x in range(1, image.width - 1):
            a = image_pixels[x-1][y-1]
            b = image_pixels[x][y-1]
            c = image_pixels[x+1][y-1]
            d = image_pixels[x-1][y]
            e = image_pixels[x+1][y]
            g = image_pixels[x-1][y+1]
            h = image_pixels[x][y+1]
            i = image_pixels[x+1][y+1]
            gradient = Sobel3_operator(a, b, c, d, e, g, h, i)
            gradients[-1].append(sqrt(gradient[0] ** 2 + gradient[1] ** 2))
            sob3_roberts_image_pixels[-1].append(one_byte_projection(
                Roberts_operator(gradient[0], gradient[1]), 0, 1))

    # finding sobel3 image
    max_grad = max([max(row) for row in gradients])
    min_grad = min([min(row) for row in gradients])
    for y in range(0, len(gradients)):
        sob3_image_pixels.append([])
        for x in range(0, len(gradients[y])):
            sob3_image_pixels[-1].append(one_byte_projection(
                gradients[y][x], min_grad, max_grad))

    return {"sob3": Image.fromarray(np.array(sob3_image_pixels, dtype=np.uint8), mode="L"),
            "sob3_roberts": Image.fromarray(np.array(sob3_roberts_image_pixels, dtype=np.uint8), mode="L")}


if __name__ == "__main__":
    try:
        filename = 'cameraman.tif'
        filefolder = 'assets/'
        filepath = path.abspath(f'{filefolder + filename}')
        with Image.open(filepath) as im:
            sob3_ims = Sobel3_operator_boundaries(im)
            for key, item in sob3_ims.items():
                item.save(path.abspath(
                    f'{filefolder + key + filename}'), mode='L')

    except Exception as e:
        print('Error occured:', e, sep='\n')
