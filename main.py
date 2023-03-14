from os import path
from PIL import Image
from calculations import Sobel_operator_boundaries, image_preparation, threshhold_operator
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

            # Draw images histogram
            draw_histogram(im.histogram(), filename)
            draw_histogram(
                sob3_ims["sob3_negative_"].histogram(), f'sob3_negative_{filename}')

            # Threshhold on sobel3_negative
            threshhold_sob3_negative_im = threshhold_operator(
                sob3_ims["sob3_negative_"], 0, 210)

            # Preparate images
            preparated1_im = image_preparation(im, 80, 200)
            preparated2_im = image_preparation(im, 5, 30)

            # Generating images with preparated images boundaries
            preparated1_sob3_ims = Sobel_operator_boundaries(
                3, preparated1_im, "preparated1_")
            preparated2_sob3_ims = Sobel_operator_boundaries(
                3, preparated2_im, "preparated2_")

            # Draw preparated2_sob3_negative image histogram
            draw_histogram(
                preparated2_sob3_ims["preparated2_sob3_negative_"].histogram(), f'preparated2_sob3_negative_{filename}')

            # Threshhold on preparated2_sobel3_negative
            threshhold_preparated2_sob3_negative_im = threshhold_operator(
                preparated2_sob3_ims["preparated2_sob3_negative_"], 0, 235)

            # Saving images
            for ims_dict in [sob3_ims, sob5_ims, preparated1_sob3_ims, preparated2_sob3_ims]:
                for key, item in ims_dict.items():
                    item.save(path.abspath(
                        f'{filefolder + key + filename}'), mode='L')
            threshhold_sob3_negative_im.save(path.abspath(
                f'{filefolder + "threshhold_sob3_negative_" + filename}'), mode="L")
            preparated1_im.save(path.abspath(
                f'{filefolder + "preparated1_" + filename}'), mode='L')
            preparated2_im.save(path.abspath(
                f'{filefolder + "preparated2_" + filename}'), mode='L')
            threshhold_preparated2_sob3_negative_im.save(path.abspath(
                f'{filefolder + "threshhold_preparated2_sob3_negative_" + filename}'), mode="L")

    except Exception as e:
        print('Error occured:', e, sep='\n')
