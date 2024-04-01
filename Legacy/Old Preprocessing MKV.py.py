import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from skimage import exposure, morphology, filters, feature
import cv2

def conv(m1, m2):
    return sum(sum(m1 * m2))

def conv2d(image, kernel):
    output = np.zeros_like(image)
    image_padded = np.pad(image, pad_width=((kernel.shape[0]//2, kernel.shape[0]//2), (kernel.shape[1]//2, kernel.shape[1]//2)), mode='constant', constant_values=0)
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x] = conv(image_padded[y:y+kernel.shape[0], x:x+kernel.shape[1]], kernel)
    return output

def enhance_bones_refined(image_path):
    """Applies refined operations to enhance bone details in an X-ray image."""
    # Open and convert to grayscale
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    
    # Apply refined CLAHE
    np_image = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))  # Fine-tuned parameters
    clahe_image = clahe.apply(np_image)
    
    # Apply Gaussian Blur with refined parameters
    blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)  # Adjust kernel size and sigma
    
    # Edge Detection
    edges = feature.canny(blurred_image.astype('uint8'), sigma=1)  # Adjust sigma as needed
    
    # Convert edges to uint8 for morphological operations
    edges_uint8 = (edges * 255).astype(np.uint8)
    
    # Apply morphological closing on edges
    selem = morphology.disk(3)  # Adjust disk size as needed
    closed_edges = morphology.closing(edges_uint8, selem)
    
    # Adaptive Thresholding on the closed edges
    adaptive_thresh = cv2.adaptiveThreshold(closed_edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Convert adaptive threshold image back to PIL image for display or saving
    final_image = Image.fromarray(adaptive_thresh)
    return final_image

# Example usage
if __name__ == "__main__":
    image_path = 'image1_502_png.rf.74e527682fa85c69a886e3069764ceed.jpg' # Replace with your actual image path
    enhanced_image = enhance_bones_refined(image_path)
    enhanced_image.show()
