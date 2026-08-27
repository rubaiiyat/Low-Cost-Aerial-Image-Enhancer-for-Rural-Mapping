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
    denoised_image = cv2.medianBlur(noisy_image, 5)

    noisy_psnr = cv2.PSNR(image_rgb, noisy_image)
    denoised_psnr = cv2.PSNR(image_rgb, denoised_image)

   
    gaussian_denoised = cv2.GaussianBlur(noisy_image, (5, 5), 0)

 
    bilateral_denoised = cv2.bilateralFilter(noisy_image, 9, 75, 75)

   
    median_psnr = cv2.PSNR(image_rgb, denoised_image)
    gaussian_psnr = cv2.PSNR(image_rgb, gaussian_denoised)
    bilateral_psnr = cv2.PSNR(image_rgb, bilateral_denoised)

    print("PSNR of noisy image:", noisy_psnr)
    print("PSNR of median filter:", median_psnr)
    print("PSNR of gaussian filter:", gaussian_psnr)
    print("PSNR of bilateral filter:", bilateral_psnr)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image)
    plt.title("Noisy Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image)
    plt.title("Denoised Image")
    plt.axis("off")

    plt.show()