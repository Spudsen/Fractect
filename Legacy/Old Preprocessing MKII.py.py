import numpy as np
from PIL import Image


def conv2d(image: np.array, kernel: np.array) -> np.array:
    f_width, f_height = kernel.shape
    conv_array = np.zeros((image.shape[0]+1 - f_width, image.shape[1]+1 - f_height))
    c_width, c_height = conv_array.shape

    for x in range(c_width):
        for y in range(c_height):
            targert = image[x:x+f_width, y:y+f_height]
            conv_array[x, y] = np.sum(conv(targert, kernel))
    return conv_array

def conv(target, filter):
    output = np.zeros(target.shape)
    for x in range(output.shape[1]):
        for y in range(output.shape[1]):
            output[x,y] = target[x,y] * filter[x,y]
    return output

image ='image1_502_png.rf.74e527682fa85c69a886e3069764ceed.jpg'