# SwiftInfer

An end-to-end image classification inference pipeline built with PyTorch, ONNX, and FastAPI.

Trains a custom CNN on CIFAR-10, exports to ONNX for optimized inference, and serves predictions via a REST API.

---

## Results

| Metric | Value |
|---|---|
| Dataset | CIFAR-10 (60,000 images, 10 classes) |
| Validation Accuracy | 71.5% |
| PyTorch Inference | 4.14 ms / image |
| ONNX Inference | 1.05 ms / image |
| Speedup | **3.94× faster** |

---

## Architecture
Input (3 × 32 × 32)
↓
Conv Block 1: Conv2d(3→32) → BatchNorm → ReLU → Conv2d(32→32) → BatchNorm → ReLU → MaxPool → Dropout
↓
Conv Block 2: Conv2d(32→64) → BatchNorm → ReLU → Conv2d(64→64) → BatchNorm → ReLU → MaxPool → Dropout
↓
Classifier: Linear(4096→512) → ReLU → Dropout → Linear(512→10)
↓
10 Class Scores

---
## Pipeline
train.py → cifar10_cnn.pth → export_onnx.py → cifar10.onnx → app/main.py → /predict
(PyTorch)      (checkpoint)     (ONNX export)    (optimized)    (FastAPI)    (REST API)

---

## Project Structure
SwiftInfer/
├── models/
│   ├── cifar10_cnn.pth       # PyTorch checkpoint
│   └── cifar10.onnx          # Exported ONNX model
├── data/                     # CIFAR-10 dataset (auto-downloaded)
├── app/
│   └── main.py               # FastAPI inference server
├── train.py                  # CNN training script
├── export_onnx.py            # ONNX export + benchmark
├── test_predict.py           # API test script
├── requirements.txt          # Dependencies
└── README.md

---

## Stack

- **PyTorch** (CPU) — model training
- **ONNX + ONNX Runtime** — model export and optimized inference
- **FastAPI + uvicorn** — REST API serving
- **Pillow + NumPy** — image preprocessing

---

## Setup

```bash
# clone the repo
git clone https://github.com/YOUR_USERNAME/SwiftInfer.git
cd SwiftInfer

# create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the model
```bash
python train.py
```
Trains for 10 epochs on CIFAR-10. Dataset downloads automatically. Saves best checkpoint to `models/cifar10_cnn.pth`.

### 2. Export to ONNX and benchmark
```bash
python export_onnx.py
```
Exports to `models/cifar10.onnx` and benchmarks PyTorch vs ONNX inference latency.

### 3. Start the API server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API
```bash
python test_predict.py
```

### 5. Interactive API docs
Open `http://localhost:8000/docs` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info and metrics |
| GET | `/health` | Health check |
| POST | `/predict` | Classify an image |

### /predict request
```json
{
  "image_b64": "<base64 encoded image string>"
}
```

### /predict response
```json
{
  "predicted_class": "cat",
  "confidence": 0.7753,
  "all_scores": {
    "cat": 0.7753,
    "dog": 0.1729,
    "frog": 0.0259,
    ...
  }
}
```

---

## Classes

`plane` `car` `bird` `cat` `deer` `dog` `frog` `horse` `ship` `truck`

---

## Why ONNX?

PyTorch carries full training machinery — autograd, gradient tracking, Python overhead. ONNX Runtime strips all of that and applies inference-specific optimizations:

- **Operator fusion** — Conv + BatchNorm + ReLU merged into a single kernel call
- **Constant folding** — fixed computations pre-calculated at export time
- **Optimized threading** — oneDNN/MKL-DNN for CPU math

Result: **3.94× speedup** with identical outputs (max difference < 1e-4).

---

*Built as a demonstration of the full ML inference pipeline — train, export, optimize, serve.*