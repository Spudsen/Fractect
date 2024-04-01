import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters

def conv(m1, m2):
    return sum(sum(m1 * m2))

def conv2d(image, kernel):
    output = np.zeros_like(image)
    image_padded = np.pad(image, pad_width=((kernel.shape[0]//2, kernel.shape[0]//2), (kernel.shape[1]//2, kernel.shape[1]//2)), mode='constant')
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x] = conv(image_padded[y: y+kernel.shape[0], x: x+kernel.shape[1]], kernel)
    return output

# Sobel operator kernels
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Open the image file
img = Image.open('image1_502_png.rf.74e527682fa85c69a886e3069764ceed.jpg')


# Convert the image to grayscale and into a numpy array
img = np.array(img.convert('L'))

# Normalize the image to 0-1
img = img / 255.0

# Apply a Gaussian blur for noise reduction
img = filters.gaussian(img, sigma=1)

# Apply the Sobel operator
result_x = conv2d(img, sobel_x)
result_y = conv2d(img, sobel_y)

# Combine the results of the Sobel operator (this is an approximation of the gradient magnitude)
result = np.hypot(result_x, result_y)
result = result / result.max()  # normalize to range 0-1

# Display the output image
plt.imshow(result, cmap='gray')
plt.show()