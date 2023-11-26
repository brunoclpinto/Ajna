from picamera2 import Picamera2
import libcamera
import cv2
import numpy as np
import time

# Initialize Picamera2
picam2 = Picamera2()

# Configure the camera
camera_config = picam2.create_video_configuration()
camera_config['main']['size'] = (640, 480)
camera_config['transform'] = libcamera.Transform(hflip=1, vflip=1)
picam2.configure(camera_config)

# Start the camera
picam2.start()

start_time = time.time()
frame_count = 0
while frame_count != 300:
    # Capture the frame
    frame = picam2.capture_array()
    frameType = type(frame)

    # Convert the frame to a format suitable for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Save the frame at regular intervals or based on a condition
    filename = f"frame_{frame_count}.jpg"
    #cv2.imwrite(filename, frame)
    print(f"Saved {filename} with content {frameType}")

    frame_count += 1

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Took {elapsed_time} to process, including saving, 300 frames which gives us {300/elapsed_time}")

picam2.stop()

