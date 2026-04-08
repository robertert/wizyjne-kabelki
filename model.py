import cv2
import numpy as np
import matplotlib.pyplot as plt

def predict(image):
    """Simple thresholding segmentation model.

    Args:
        image: numpy array of shape (H, W, 3), uint8 RGB image.

    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
    """


    gray = np.mean(image, axis=2)
    mask = (gray > 128).astype(np.uint8) * 255
    return mask


if __name__ == "__main__":
    # Example usage
    image = cv2.imread("example_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = predict(image)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.show()