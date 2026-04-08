import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

def predict(image):
    """Simple thresholding segmentation model using Saturation channel.

    Args:
        image: numpy array of shape (H, W, 3), uint8 RGB image.

    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
    """
    # Convert RGB to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract ONLY the Saturation (S) channel
    # White casing and dark background have very low saturation (close to 0).
    # Colored insulators and copper have high saturation.
    saturation = hsv[:, :, 1]

    # Apply CLAHE to the saturation channel to boost the colors, especially copper
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sat_clahe = clahe.apply(saturation)

    # Apply stronger blurring to remove background noise
    blurred = cv2.GaussianBlur(sat_clahe, (11, 11), 0)

    # Threshold using Otsu's method on the Saturation channel
    # This automatically ignores the white casing and background.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- BORDER CLEARING ---

    # Create marker from image borders
    marker = np.zeros_like(mask)
    marker[0, :] = mask[0, :]
    marker[-1, :] = mask[-1, :]
    marker[:, 0] = mask[:, 0]
    marker[:, -1] = mask[:, -1]

    # Morphological reconstruction (iterative dilation)
    kernel_small = np.ones((3, 3), np.uint8)

    prev = np.zeros_like(marker)
    curr = marker.copy()

    while True:
        dilated = cv2.dilate(curr, kernel_small)
        curr = np.minimum(dilated, mask)
        
        if np.array_equal(curr, prev):
            break
        prev = curr.copy()

    # Remove border-connected objects
    mask = cv2.subtract(mask, curr)

    return mask


if __name__ == "__main__":
    # Define the output directory and create it if it doesn't exist
    output_dir = "./cable/train/good_masks"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(224):
        filename = f"{i:03d}.png"
        
        input_path = f"./cable/train/good/{filename}"
        output_path = f"{output_dir}/{filename}"
        
        image_bgr = cv2.imread(input_path)
        
        if image_bgr is not None:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            mask = predict(image_rgb)
            
            cv2.imwrite(output_path, mask)
        else:
            print(f"Warning: Missing file {input_path}")

        if i == 0:
            break
        #     # Display the first image and its mask for verification
        #     plt.figure(figsize=(10, 5))
        #     plt.subplot(1, 2, 1)
        #     plt.title("Input Image")
        #     plt.imshow(image_rgb)
        #     plt.axis('off')

        #     plt.subplot(1, 2, 2)
        #     plt.title("Predicted Mask")
        #     plt.imshow(mask, cmap='gray')
        #     plt.axis('off')

        #     plt.show()