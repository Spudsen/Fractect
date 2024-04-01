import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from skimage import exposure, morphology, filters, feature
import cv2

def preprocess_image(image_path, model_input_size=(416, 416)):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize the image with padding to maintain aspect ratio
    h, w, _ = image.shape
    scale = min(model_input_size[0] / h, model_input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Create a new image with the specified size and fill with gray (128)
    new_image = np.full((model_input_size[0], model_input_size[1], 3), 128, dtype=np.uint8)
    # Copy the resized image to the center of the new image
    y_offset, x_offset = (model_input_size[0] - nh) // 2, (model_input_size[1] - nw) // 2
    new_image[y_offset:y_offset+nh, x_offset:x_offset+nw, :] = image_resized
    
    # Normalize the image
    image_normalized = new_image / 255.0
    
    return image_normalized

# Example usage
image_path = 'image3_278_png.rf.7799f27562a3d9615702eebad0f44bfd.jpg'
processed_image = preprocess_image(image_path, model_input_size=(200, 200))
print(processed_image.shape)  # Should print (100, 100, 3) for this example

# Display the processed image
plt.imshow(processed_image)
plt.show()