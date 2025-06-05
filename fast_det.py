
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the YOLO model 
model = YOLO("best.pt") 

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # read the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(image)

    # predict
    results = model(image_array)[0]

    # Draw the squares on the image
    for box in results.boxes.data:
        x1, y1, x2, y2, score, class_id = box.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = model.names[int(class_id)]
        cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_array, f"{label} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convet image to bytes
    _, img_encoded = cv2.imencode(".jpg", image_array)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")







   


