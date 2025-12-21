import cv2
import numpy as np

img = cv2.imread('sonet.png', 0)

# 1. Blurring
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 2. Adaptive Binarization
binary = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# 3. Noise Cleaning
clean_result = cv2.medianBlur(binary, 3)

cv2.imwrite('task1_result.png', clean_result)
print("Task 1 Done.")


img_noisy = cv2.imread('sonet_noised.png', 0)

# 1. Stronger Blurring
blurred_noisy = cv2.medianBlur(img_noisy, 5)

# 2. Adaptive Binarization
binary_noisy = cv2.adaptiveThreshold(blurred_noisy, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 5)

# 3. Noise Cleaning
kernel = np.ones((3,3), np.uint8)
clean_noisy_result = cv2.morphologyEx(binary_noisy, cv2.MORPH_OPEN, kernel)

cv2.imwrite('task2_result.png', clean_noisy_result)
print("Task 2 Done.")