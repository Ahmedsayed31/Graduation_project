from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
from PIL import Image
import io
import onnxruntime as ort
from ultralytics import YOLO
import cv2

app = FastAPI()

# ---- Load the classification ONNX model ---- #
classification_model_path = "models/model_vgg_lv.onnx"
session = ort.InferenceSession(classification_model_path)

# Classes
classes = {0:'Cyst', 1:'Normal', 2:'Stone', 3:'Tumor'}
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---- Detection model is not loaded at start ---- #
yolo_model = None  # Lazy load when needed

# Utility function to preprocess image
def preprocess_image(file: UploadFile):
    image_file = io.BytesIO(file)
    image = Image.open(image_file).convert("RGB")
    img = image.resize((225, 225))
    img_arr = np.array(img) / 255.0
    final_img = np.expand_dims(img_arr, axis=0).astype(np.float32)
    return image, final_img

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image, image_array = preprocess_image(contents)

        # ---- Step 1: Classification ---- #
        outputs = session.run([output_name], {input_name: image_array})
        predictions = outputs[0][0]
        predicted_class = int(np.argmax(predictions))
        class_label = classes[predicted_class]

        result = {"classification_result": class_label}

        # ---- Step 2: Detection if Tumor or Stone ---- #
        if class_label in ["Tumor", "Stone"]:
            global yolo_model
            if yolo_model is None:
                yolo_model = YOLO("models/best.pt")  # Load YOLO only if needed

            original_image = np.array(pil_image)
            results = yolo_model(original_image)[0]

            # Draw bounding boxes
            for box in results.boxes.data:
                x1, y1, x2, y2, score, class_id = box.tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = yolo_model.names[int(class_id)]
                cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_image, f"{label} {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Encode image for response
            _, img_encoded = cv2.imencode(".jpg", original_image)
            img_bytes = io.BytesIO(img_encoded.tobytes())

            # Return classification and detection image
            return StreamingResponse(img_bytes, media_type="image/jpeg", headers={"classification": class_label})

        else:
            # Just return classification if Normal or Cyst
            return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
