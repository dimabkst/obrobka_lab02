from os import path
from PIL import Image
from calculations import Sobel_operator_boundaries, image_preparation
from utils import draw_histogram


if __name__ == "__main__":
    try:
        filename = 'cameraman.tif'
        filefolder = 'assets/'
        filepath = path.abspath(f'{filefolder + filename}')

        with Image.open(filepath) as im:
            # Generating images with image boundaries
            sob3_ims = Sobel_operator_boundaries(3, im)
            sob5_ims = Sobel_operator_boundaries(5, im)

            # # Draw image histogram
            # draw_histogram(im.histogram(), filename)

            # Preparate image
            preparated_im = image_preparation(im, 80, 200)

            # Generating images with preparated image boundaries
            preparated_sob3_ims = Sobel_operator_boundaries(
                3, preparated_im, "preparated_")

            # Saving images
            for ims_dict in [sob3_ims, sob5_ims, preparated_sob3_ims]:
                for key, item in ims_dict.items():
                    item.save(path.abspath(
                        f'{filefolder + key + filename}'), mode='L')
            preparated_im.save(path.abspath(
                f'{filefolder + "preparated_" + filename}'), mode='L')

    except Exception as e:
        print('Error occured:', e, sep='\n')
