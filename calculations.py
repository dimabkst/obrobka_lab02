from typing import Tuple, Dict, List
from PIL import Image
from math import sqrt
import numpy as np


def one_byte_projection(old_intensity: float, min_old_intensity: float, max_old_intensity: float) -> int:
    return 255 * (old_intensity - min_old_intensity) // (max_old_intensity - min_old_intensity)


def Sobel3_operator(f3x3: List[List[int]]) -> Tuple[int, int]:
    # actually Gy, used name from lab file
    Gx = (f3x3[2][0] + f3x3[2][2] - f3x3[0][0] - f3x3[0][2]) +\
        (f3x3[2][1] - f3x3[0][1]) * 2
    Gy = (f3x3[0][2] + f3x3[2][2] - f3x3[0][0] - f3x3[2][0]) +\
        (f3x3[2][1] - f3x3[0][1]) * 2  # actually Gx

    return (Gx, Gy)


def Sobel5_operator(f5x5: List[List[int]]) -> Tuple[float, float]:
    # actually Gy, used name from lab file
    Gx = (f5x5[0][0] + f5x5[0][4] - f5x5[4][0] - f5x5[4][4]) / 4 +\
        (f5x5[0][1] + f5x5[0][3] + f5x5[1][0] + f5x5[1][4] - f5x5[3][0] - f5x5[3][4] - f5x5[4][1]-f5x5[4][3]) / 2 +\
        (f5x5[0][2] + f5x5[1][1] + f5x5[1][3] - f5x5[3][1] - f5x5[3][3] - f5x5[4][2]) +\
        (f5x5[1][2] - f5x5[3][2]) * 2
    Gy = (f5x5[0][4] + f5x5[4][4] - f5x5[0][0] - f5x5[4][0]) / 4 +\
        (f5x5[0][3] + f5x5[1][4] + f5x5[3][4] + f5x5[4][3] - f5x5[0][1] - f5x5[1][0] - f5x5[3][0]-f5x5[4][1]) / 2 +\
        (f5x5[1][3] + f5x5[2][4] + f5x5[3][3] - f5x5[1][1] - f5x5[2][0] - f5x5[3][1]) +\
        (f5x5[2][3] - f5x5[2][1]) * 2  # actually Gx

    return (Gx, Gy)


def Sobel_operator(dim: int, fs: List[List[int]]) -> Tuple[float, float]:
    Sobel_operators = {
        3: Sobel3_operator,
        5: Sobel5_operator,
    }

    if dim not in Sobel_operators.keys():
        raise Exception(f'Sobel {dim} x {dim} operator is not implemeted')

    # check whether fs is really dim x dim list
    if len(fs) != dim:
        raise Exception(f'Invalid Sobel {dim} x {dim} operator input')
    for row in fs:
        if len(row) != dim:
            raise Exception(f'Invalid Sobel {dim} x {dim} operator input')

    return Sobel_operators[dim](fs)


def Sobel_operator_boundaries(dim: int, image: Image.Image, dict_key_prefix: str = '') -> Dict[Image.Image, Image.Image]:
    image_pixels = [[image.getpixel((i, j)) for j in range(
        image.width)] for i in range(image.height)]  # save it, so can use more efficiently in next loop
    op_shift = (dim - 1) // 2

    # finding gradients
    gradients = []
    for y in range(1, image.height - op_shift):
        gradients.append([])
        for x in range(1, image.width - op_shift):
            gradient = Sobel_operator(dim,
                                      [[image_pixels[i][j] for j in range(y - op_shift, y + op_shift + 1)]
                                       for i in range(x - op_shift, x + op_shift + 1)])
            gradients[-1].append(sqrt(gradient[0] ** 2 + gradient[1] ** 2))

    # finding sobel (and negative) image
    sob_image_pixels = []
    sob_negative_image_pixels = []
    max_grad = max([max(row) for row in gradients])
    min_grad = min([min(row) for row in gradients])
    for y in range(0, len(gradients)):
        sob_image_pixels.append([])
        sob_negative_image_pixels.append([])
        for x in range(0, len(gradients[y])):
            pixel_byte = one_byte_projection(
                gradients[y][x], min_grad, max_grad)
            sob_image_pixels[-1].append(pixel_byte)
            sob_negative_image_pixels[-1].append(one_byte_projection(
                max_grad, min_grad, max_grad) - pixel_byte)

    return {
        f'{dict_key_prefix}sob{dim}_': Image.fromarray(np.array(sob_image_pixels, dtype=np.uint8), mode="L"),
        f'{dict_key_prefix}sob{dim}_negative_': Image.fromarray(np.array(sob_negative_image_pixels, dtype=np.uint8), mode="L"),
    }


# preparation mentioned in lab spec
def image_preparation(image: Image.Image, min_brightness: int, max_brightness: int) -> Image.Image:
    preparated_image_pixels = []
    for y in range(image.height):
        preparated_image_pixels.append([])
        for x in range(image.width):
            pixel_brigtness = image.getpixel((x, y))
            if pixel_brigtness < min_brightness or pixel_brigtness > max_brightness:
                pixel_brigtness = min_brightness  # so after one_byte_projection it would be 0
            preparated_image_pixels[-1].append(one_byte_projection(
                pixel_brigtness, min_brightness, max_brightness))

    return Image.fromarray(np.array(preparated_image_pixels, dtype=np.uint8), mode="L")


def threshhold_operator(image: Image.Image, min_brightness: int = 0, max_brightness: int = 255) -> Image.Image:
    changed_image_pixels = []
    for y in range(image.height):
        changed_image_pixels.append([])
        for x in range(image.width):
            pixel_brigtness = image.getpixel((x, y))
            if pixel_brigtness < min_brightness:
                pixel_brigtness = 0
            elif pixel_brigtness > max_brightness:
                pixel_brigtness = 255
            changed_image_pixels[-1].append(pixel_brigtness)

    return Image.fromarray(np.array(changed_image_pixels, dtype=np.uint8), mode="L")
