import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("dataset/rural_area.jpg")

if image is None:
    print("Image not found!")
else:
    print("Image loaded successfully!")
    print("Image size:", image.shape)


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

   
    noise = np.random.normal(0, 25, image_rgb.shape)

    noisy_image = image_rgb + noise

    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image)
    plt.title("Noisy Image")
    plt.axis("off")

    plt.show()