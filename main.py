import cv2
import matplotlib.pyplot as plt

image = cv2.imread("dataset/rural_area.jpg")

if image is None:
    print("Image not found!")
else:
    print("Image loaded successfully!")
    print("Image size:", image.shape)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()