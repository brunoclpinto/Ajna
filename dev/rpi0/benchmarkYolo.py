import numpy as np
from PIL import Image
import time
from tflite_runtime.interpreter import Interpreter

def inference_tflite(model_path, input_data):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])
    
def preprocess_image(image):
    #image = image.resize((1024, 1024))
    input_array = np.array(image, dtype=np.float32)
    input_array = input_array / 255.0
    input_array = np.expand_dims(input_array, axis=0)
    return input_array

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])
    
def processModelResults(output_data):
    scaling_factors = np.array([original_image.width, original_image.height, original_image.width, original_image.height])
    scaling_factors_reshaped = scaling_factors.reshape(1, 4, 1)

    boxes = output_data[:, :4] * scaling_factors_reshaped
    boxes = boxes[0].T
    scores = output_data[:, 4]
    scores = scores[0]
    boxes = boxes[scores >= 0.75]
    return np.hstack([(boxes[:, 0] - 0.5 * boxes[:, 2]).reshape(-1, 1),
                       (boxes[:, 1] - 0.5 * boxes[:, 3]).reshape(-1, 1),
                       (boxes[:, 0] + 0.5 * boxes[:, 2]).reshape(-1, 1),
                       (boxes[:, 1] + 0.5 * boxes[:, 3]).reshape(-1, 1)])

def save_detected_boxes(xo, yo, xm, ym):
    cropped_image = original_image.crop((xo, yo, xm, ym))
    file_name = f"face_box_{time.time()}.jpg"
    cropped_image.save(f"test-dev/testResults/{file_name}")

if __name__ == "__main__":
    model_path = "models/yolov8n-640x480.tflite"
    img_path = "testMedia/face640x480.jpg"
    
    start_time = startStep = time.time()
    original_image = Image.open(img_path)    
    input_data = preprocess_image(original_image)
    end_time = time.time()
    elapsed_time = end_time - startStep
    print(f"Image processing in: {elapsed_time} seconds")    
    
    for _ in range(10):
        startStep = time.time()
        outputData = inference_tflite(model_path, input_data)
        end_time = time.time()
        elapsed_time = end_time - startStep
        print(f"Inference in: {elapsed_time} seconds")    

        startStep = time.time()
        boxes = processModelResults(outputData)
        end_time = time.time()
        elapsed_time = end_time - startStep
        print(f"Inference result processed in: {elapsed_time} seconds")    

        startStep = time.time()
        vectorized_function = np.vectorize(save_detected_boxes)
        vectorized_function(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
        end_time = time.time()
        elapsed_time = end_time - startStep
        print(f"Filtered content and cropped/saved in: {elapsed_time} seconds")    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds to detect {len(boxes)}")