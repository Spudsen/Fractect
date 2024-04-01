import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from skimage import exposure

def conv(m1, m2):
    return sum(sum(m1 * m2))

def conv2d(image, kernel):
    output = np.zeros_like(image)
    image_padded = np.pad(image, pad_width=((kernel.shape[0]//2, kernel.shape[0]//2), (kernel.shape[1]//2, kernel.shape[1]//2)), mode='constant')
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x] = conv(image_padded[y: y+kernel.shape[0], x: x+kernel.shape[1]], kernel)
    return output

def histogram_equalization(image_path):
    """Applies histogram equalization to an image."""
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    equalized_image = ImageOps.equalize(image)
    return equalized_image

def clahe(image_path):
    """Applies CLAHE to an image."""
    image = Image.open(image_path)
    image = np.array(ImageOps.grayscale(image))
    clahe_image = exposure.equalize_adapthist(image, clip_limit=0.03)
    return Image.fromarray((clahe_image * 255).astype(np.uint8))

# Example usage
if __name__ == "__main__":
    image_path = 'image1_502_png.rf.74e527682fa85c69a886e3069764ceed.jpg'
    
    # Apply Histogram Equalization
    hist_eq_image = histogram_equalization(image_path)
    hist_eq_image.show()
    
    # Apply CLAHE
    clahe_image = clahe(image_path)
    clahe_image.show()