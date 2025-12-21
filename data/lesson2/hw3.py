import cv2
import numpy as np

image_path = 'darken.png'
img = cv2.imread(image_path)

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)

# Apply equalization to the V (Value/Brightness) channel
v_eq = cv2.equalizeHist(v)

hsv_eq = cv2.merge([h, s, v_eq])
result_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

v_float = v.astype(np.float32)

# Increase brightness by 50%
v_scaled = v_float * 1.5

# Clip values to the [0, 255] range to avoid overflow and cast back to uint8
v_clipped = np.clip(v_scaled, 0, 255)
v_final_2 = v_clipped.astype(np.uint8)

hsv_scaled = cv2.merge([h, s, v_final_2])
result_scaled = cv2.cvtColor(hsv_scaled, cv2.COLOR_HSV2BGR)

cv2.imshow('Original', img)
cv2.imshow('Histogram Equalization', result_eq)
cv2.imshow('Scaled Brightness (+50%)', result_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()