import io
from PIL import Image
import numpy as np
import onnxruntime as ort
import warnings
import cv2
from ultralytics import YOLO
warnings.filterwarnings("ignore")

# Load the model
model_path = "models/model_vgg_lv.onnx"
session = ort.InferenceSession(model_path)

# Get the input and output names to use in the inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = {0:'Cyst', 1:'Normal',2:'Stone',3:"Tumor"}

def preprocess_image(file):
    # Read and process the uploaded image
    image_file = io.BytesIO(file.read())
    image = Image.open(image_file).convert("RGB")  # Ensure image is in RGB format
        
    # Resize and normalize the image
    img = image.resize((225, 225))  # Resize to match the model's input size
    img_arr = np.array(img) / 255.0  # Normalize pixel values
    final_img = np.expand_dims(img_arr, axis=0).astype(np.float32)  # Add batch dimension and ensure float32 type

    # Perform inference using the ONNX model
    outputs = session.run([output_name], {input_name: final_img})

    # Process the predictions
    predictions = outputs[0][0]  # Extract the prediction array
    predicted_class = int(np.argmax(predictions))  # Get the index of the highest probability
    result = classes[predicted_class]

    return result , image


def load_model():
    model = YOLO('models/best.pt')
    return model



def detect_disease(image):
    """Detect diseases using YOLO"""
    image_array = np.array(image)

    model = load_model()
    results = model(image_array)[0]
    
    # Draw detections
    for box in results.boxes.data:
        x1, y1, x2, y2, score, class_id = box.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = model.names[int(class_id)]
        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_array, f"{label} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return Image.fromarray(image_array), results


# model = load_model()