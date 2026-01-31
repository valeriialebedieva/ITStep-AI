import cv2
import numpy as np
from ultralytics import YOLO

img_path = 'tumor1.jpg'
img = cv2.imread(img_path)

model_path = 'brain-tumor-seg.pt'
model = YOLO(model_path)

results = model.predict(source=img, save=False, verbose=False)

if results[0].masks is not None:
    mask_raw = results[0].masks.data[0].cpu().numpy()
    mask = (mask_raw * 255).astype("uint8")

    h, w = img.shape[:2]
    mask = cv2.resize(mask, (w, h))

    area_pixels = cv2.countNonZero(mask)
    pixel_factor = 0.0025
    area_real = area_pixels * pixel_factor

    print(f"Area in pixels: {area_pixels}")
    print(f"Real area: {area_real:.2f}")

    if area_real < 10:
        tumor_type = "small"
    elif 10 <= area_real <= 25:
        tumor_type = "middle"
    else:
        tumor_type = "large"

    print(f"Tumor type: {tumor_type}")

    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(tumor_type, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No tumor detected")