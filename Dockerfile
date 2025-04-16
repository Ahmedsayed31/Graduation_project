FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --default-timeout=500

RUN pip install --no-cache-dir gdown

RUN gdown --id 1DotvCz1zV57eZ--TeoSrNLUcMZho33JE -O Kidney_model.onnx

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
