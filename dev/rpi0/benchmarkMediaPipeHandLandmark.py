from picamera2 import Picamera2
import libcamera
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def startCamera():
    # Initialize Picamera2
    picam2 = Picamera2()
    
    # Configure the camera
    camera_config = picam2.create_video_configuration()
    camera_config['main']['size'] = (640, 480)
    camera_config['transform'] = libcamera.Transform(hflip=1, vflip=1)
    picam2.configure(camera_config)

    # Start the camera
    picam2.start()

    return picam2

def stopCamera(camera):
    camera.stop()

def getFrame(camera):
    frame = camera.capture_array("main")

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

class landmarker_and_result():
   def __init__(self):
      self.result = mp.tasks.vision.HandLandmarkerResult
      self.landmarker = mp.tasks.vision.HandLandmarker
      self.createLandmarker()
   
   def createLandmarker(self):
      # callback function
      def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
         self.result = result

      # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
      options = mp.tasks.vision.HandLandmarkerOptions( 
         base_options = mp.tasks.BaseOptions(model_asset_path="models/mediaPipeHandLandmarks.task"), # path to model
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         num_hands = 2, # track both hands
         min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
         min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
         min_tracking_confidence = 0.3, # lower than value to get predictions more often
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   def detect_async(self, frame):
      # convert np frame to mp image
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # detect landmarks
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   def close(self):
      # close landmarker
      self.landmarker.close()


print("Gonna start camera")
camera = startCamera()
print("camera started, load detector")
detector = landmarker_and_result()
print("detector loaded")

start_time = time.time()
maxFrames = 1200
for _ in range(maxFrames):
    stepStartTime = time.time()
    print("lets detect")
    detector.detect_async(getFrame(camera))
    print(detector.result)
    print("detected")
    end_time = time.time()
    elapsed_time = end_time - stepStartTime
    print(f"Took {elapsed_time} to process a frame")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Took {elapsed_time} to process, including detection, {maxFrames} frames which gives us {maxFrames/elapsed_time}")

detector.close()
stopCamera(camera)
