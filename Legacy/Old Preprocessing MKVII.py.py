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

def enhance_bones_with_contours(image_path):
    """Enhance bones by preserving natural curvature using contour analysis."""
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(original)
    
    # Gaussian Blur to reduce noise and preserve curves
    blurred = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)
    
    # Sobel Edge Detection
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    
    # Normalize and threshold the sobel edges
    sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(sobel_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours and create a new mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask_with_contours = np.zeros_like(mask)
    cv2.drawContours(mask_with_contours, contours, -1, (255), thickness=cv2.FILLED)
    
    # Optionally, apply morphological operations to smooth the contours
    selem = morphology.disk(1)  # Use a smaller disk to preserve curves
    mask_smoothed = morphology.opening(mask_with_contours, selem)
    
    # Apply the smoothed mask to the original image
    result = cv2.bitwise_and(original, original, mask=mask_smoothed.astype(np.uint8))
    
    # Convert the result to PIL image for display or saving
    final_image = Image.fromarray(result)
    return final_image


image_path = 'image1_502_png.rf.74e527682fa85c69a886e3069764ceed.jpg'
enhanced_image = enhance_bones_with_contours(image_path)
enhanced_image.show()


