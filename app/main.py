from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from PIL import Image
import base64
import io

# app setup
app = FastAPI(
    title="SwiftInfer",
    description="CIFAR-10 image classification via ONNX Runtime",
    version="1.0.0"
)

# constants
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

ONNX_PATH = "models/cifar10.onnx"

# load ONNX session once at startup
print("Loading ONNX model...")
session = ort.InferenceSession(ONNX_PATH)
print("ONNX model loaded ✓")

# request/response schemas
class ImageInput(BaseModel):
    image_b64: str

class PredictionOutput(BaseModel):
    predicted_class: str
    confidence: float
    all_scores: dict

# preprocessing
def preprocess(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32), Image.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - MEAN) / STD
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# routes
@app.get("/")
def root():
    return {
        "service": "SwiftInfer",
        "status": "running",
        "model": "CIFAR-10 CNN via ONNX Runtime",
        "accuracy": "71.5%",
        "speedup": "3.94x over PyTorch"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: ImageInput):
    try:
        img = preprocess(payload.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Image preprocessing failed: {str(e)}")
    try:
        logits = session.run(["logits"], {"image": img})[0][0]
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Inference failed: {str(e)}")

    # softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx])
    all_scores = {CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CLASSES))}

    return PredictionOutput(
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
        all_scores=all_scores
    )