from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()

# CIFAR-10 klasser
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ladda ONNX-modellen vid start
session = ort.InferenceSession("model.onnx")


class PredictRequest(BaseModel):
    data: list[float]


class PredictResponse(BaseModel):
    prediction: str
    class_id: int
    confidence: float


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # input ska vara 3072 värden (3x32x32), normaliserade
    input_array = np.array(request.data, dtype=np.float32)
    input_array = input_array.reshape(1, 3, 32, 32)

    # kör inference
    result = session.run(None, {"image": input_array})
    logits = result[0][0]

    # softmax för confidence
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    return PredictResponse(
        prediction=CLASSES[class_id],
        class_id=class_id,
        confidence=round(confidence, 4)
    )
