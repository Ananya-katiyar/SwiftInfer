import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time
import os

# architecture (must match train.py exactly)
NUM_CLASSES = 10
MODEL_SAVE_PATH   = "models/cifar10_cnn.pth"
ONNX_SAVE_PATH    = "models/cifar10.onnx"

class SwiftInferCNN(nn.Module):
    def __init__(self):
        super(SwiftInferCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# load trained model
print("Loading trained model...")
model = SwiftInferCNN()
model.load_state_dict(torch.load(MODEL_SAVE_PATH,
                                  map_location="cpu",
                                  weights_only=True))
model.eval()
print("Model loaded ✓")

# export to ONNX 
print("\nExporting to ONNX...")
dummy_input = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_SAVE_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image":  {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)
print(f"ONNX model saved to {ONNX_SAVE_PATH} ✓")

# validate ONNX model
print("\nValidating ONNX model...")
onnx_model = onnx.load(ONNX_SAVE_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX model structure valid ✓")

# compare PyTorch vs ONNX outputs
print("\nComparing PyTorch vs ONNX outputs on same input...")
test_input = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    pytorch_out = model(test_input).numpy()

session = ort.InferenceSession(ONNX_SAVE_PATH)
onnx_out = session.run(["logits"], {"image": test_input.numpy()})[0]

max_diff = np.max(np.abs(pytorch_out - onnx_out))
print(f"Max output difference: {max_diff:.8f}")
if max_diff < 1e-4:
    print("Outputs match ✓  (difference is negligible)")
else:
    print("WARNING: outputs differ — check export settings")

# benchmark: PyTorch vs ONNX inference latency 
print("\nBenchmarking inference latency (100 runs each)...")
RUNS = 100
sample = torch.randn(1, 3, 32, 32)

# warmup
with torch.no_grad():
    for _ in range(5):
        _ = model(sample)

# pytorch timing
start = time.perf_counter()
with torch.no_grad():
    for _ in range(RUNS):
        _ = model(sample)
pytorch_ms = (time.perf_counter() - start) * 1000 / RUNS

# onnx warmup
sample_np = sample.numpy()
for _ in range(5):
    session.run(["logits"], {"image": sample_np})

# onnx timing
start = time.perf_counter()
for _ in range(RUNS):
    session.run(["logits"], {"image": sample_np})
onnx_ms = (time.perf_counter() - start) * 1000 / RUNS

speedup = pytorch_ms / onnx_ms

print(f"\n{'='*45}")
print(f"  PyTorch inference : {pytorch_ms:.2f} ms per image")
print(f"  ONNX    inference : {onnx_ms:.2f} ms per image")
print(f"  Speedup           : {speedup:.2f}x")
print(f"{'='*45}")
print(f"\nONNX model size : {os.path.getsize(ONNX_SAVE_PATH)/1024/1024:.2f} MB")
print(f"PTH  model size : {os.path.getsize(MODEL_SAVE_PATH)/1024/1024:.2f} MB")

print("\nDay 2 complete. Ready for FastAPI serving.")