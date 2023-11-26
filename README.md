# XTRAEyes
Open-source project for leveraging current AI to help the visually impaired 

# Dev
## Yolov8n ☑️
* Will be used for object detection (possibly)
* Current usable sweep spot is 640x480 
* Mac Mini M1 base model is 28x times faster in single thread than RPI Zero 2W
* Inference time at 1.96s for rpi0
* Loading image and processing model results takes 2.1s
* Requires Coral USB accelerator (eventually) 
* wihout coral its still usefull under a menu option

## MediaPipe Hands ✅
* configure to detect hand landmarks
* around 26 frames per second
* requires palm to start tracking hand

## RPI CAM V3 ✅
* configured for 640x480
* max 29 frames per second, its awesome