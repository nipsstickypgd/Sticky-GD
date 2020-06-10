from PIL import Image

from common import data_folder, temp_folder
import numpy as np


def read_image(name):
    img = Image.open(data_folder + 'images/' + name + ".png").convert('LA')
    img.save(temp_folder + name + '_gray.png')
    return np.array(img)[:, :, 0]
