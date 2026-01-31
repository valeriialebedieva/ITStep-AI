import cv2
import numpy as np

# 1. Load the image
# Make sure the path to the file is correct
image_path = 'tumor1.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not open image {image_path}")
else:
    # 2. Segmentation
    # Convert to grayscale for easier analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # VARIANT A: If you have the pre-made mask file mentioned in the task
    # mask = cv2.imread('data/lesson_seg/brain-tumor-seg.jpg', 0)

    # VARIANT B: Create the mask programmatically (since the tumor is the brightest object)
    # Apply Thresholding. Pixels brighter than 160 become white (255), others black (0).
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    # Clean up noise (erosion followed by dilation) to keep the shape solid
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 3. Calculate Area
    # Count the number of white pixels (non-zero pixels in the mask)
    area_pixels = cv2.countNonZero(mask)

    # Conversion factor: 1 pixel = 0.0025 units
    pixel_factor = 0.0025
    area_real = area_pixels * pixel_factor

    print(f"Area in pixels: {area_pixels}")
    print(f"Real area: {area_real:.2f}")

    # 4. Tumor Classification
    tumor_type = "unknown"

    if area_real < 10:
        tumor_type = "small"
    elif 10 <= area_real <= 25:
        tumor_type = "middle"
    else:
        tumor_type = "large"

    print(f"Tumor type: {tumor_type}")

    # 5. Visualize Result
    # Use bitwise_and to apply the mask to the original image
    # (Background becomes black/0, only the tumor remains visible)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Show the result. The window title is the tumor type (e.g., "large").
    cv2.imshow(tumor_type, result)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()