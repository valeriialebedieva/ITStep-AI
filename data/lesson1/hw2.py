import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image, cmap=None):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()

# 1. Load the main image
image_path = 'Lenna.png'
original_img = cv2.imread(image_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
h, w, _ = original_img.shape

mask1 = np.zeros((h, w), dtype=np.uint8)
center_coordinates = (w // 3, h // 3)
radius = 100
cv2.circle(mask1, center_coordinates, radius, 255, -1)

mask2 = np.zeros((h, w), dtype=np.uint8)
start_point = (w // 3, h // 3)
end_point = (w - 50, h - 50)
cv2.rectangle(mask2, start_point, end_point, 255, -1)


# Step 2: Combine two masks into one using cv2.bitwise_or()
combined_mask_uint8 = cv2.bitwise_or(mask1, mask2)
show_image("Combined Mask (Bitwise OR)", combined_mask_uint8, cmap='gray')

# Step 4: Change data type to bool
mask1_bool = mask1.astype(bool)
mask2_bool = mask2.astype(bool)
combined_mask_bool = combined_mask_uint8.astype(bool)


def apply_bool_mask(img, mask_bool):
    # Create a black background (zeros)
    result = np.zeros_like(img)
    # Copy pixels only where mask is True
    result[mask_bool] = img[mask_bool]
    return result

# Step 3: Display parts of the image corresponding to specific masks
res_mask1 = apply_bool_mask(original_img, mask1_bool)
show_image("Result: Mask 1 (Circle)", res_mask1)
res_mask2 = apply_bool_mask(original_img, mask2_bool)

show_image("Result: Mask 2 (Rectangle)", res_mask2)
res_combined = apply_bool_mask(original_img, combined_mask_bool)
show_image("Result: Combined Masks", res_combined)