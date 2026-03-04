from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

app = FastAPI()

# CIFAR-10 klasserna
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ladda modellen
session = ort.InferenceSession("model.onnx")


class PredictRequest(BaseModel):
    data: list[float]


@app.post("/predict")
def predict(request: PredictRequest):
    # kolla att input har rätt storlek
    if len(request.data) != 3072:
        raise HTTPException(status_code=400, detail="behöver exakt 3072 värden (3x32x32)")

    # gör om till numpy array
    input_array = np.array(request.data, dtype=np.float32)
    input_array = input_array.reshape(1, 3, 32, 32)

    # kör modellen
    result = session.run(None, {"image": input_array})
    output = result[0][0]

    class_id = int(np.argmax(output))

    return {
        "prediction": CLASSES[class_id],
        "class_id": class_id
    }
