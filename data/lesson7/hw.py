import cv2
import os

input_path = 'meter.mp4'
output_path = 'meter_binary.mp4'

if not os.path.exists(input_path):
    print(f"Error: File {input_path} not found. Please check the path.")
    exit()

cap = cv2.VideoCapture(input_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

        #  Noise Filtering (Bilateral Filter)
    filtered_frame = cv2.bilateralFilter(frame, 9, 75, 75)

    # Convert to Grayscale
    gray_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)

    # Binarization
    _, binary_frame = cv2.threshold(gray_frame, 90, 255, cv2.THRESH_BINARY)
    final_output = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)

    # Write the frame to the output file
    out.write(final_output)

cap.release()
out.release()
cv2.destroyAllWindows()

