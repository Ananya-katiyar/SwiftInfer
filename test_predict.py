import base64
import json
import urllib.request
import pickle
import numpy as np
from PIL import Image

with open("data/cifar-10-batches-py/test_batch", "rb") as f:
    data = pickle.load(f, encoding="bytes")

img_array = data[b"data"][0].reshape(3, 32, 32).transpose(1, 2, 0)
true_label = data[b"labels"][0]

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

img = Image.fromarray(img_array.astype(np.uint8))
img.save("test_image.png")
print(f"True label: {CLASSES[true_label]}")

with open("test_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = json.dumps({"image_b64": image_b64}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode("utf-8"))

print(f"Predicted class : {result['predicted_class']}")
print(f"Confidence      : {result['confidence']*100:.1f}%")
print(f"\nAll scores:")
for cls, score in sorted(result['all_scores'].items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 40)
    print(f"  {cls:<8} {score:.4f}  {bar}")