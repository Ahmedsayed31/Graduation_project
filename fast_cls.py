
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
from PIL import Image
import io
import onnxruntime as ort


model_path = "model_vgg_lv.onnx"
session = ort.InferenceSession(model_path)

app = FastAPI()

classes = {0:'Cyst', 1:'Normal',2:'Stone',3:"Tumor"}

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        image_file = io.BytesIO(await file.read())
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

        # Return the result 
        return {"Result is": classes[predicted_class]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
