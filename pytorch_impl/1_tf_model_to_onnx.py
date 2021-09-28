import subprocess

from config import MODEL_NAME, PROJECT_ROOT
from models import create_model

model = create_model()
model.save(f"{PROJECT_ROOT}/models/{MODEL_NAME}")

bashCmd = [
    "python", "-m", "tf2onnx.convert", "--saved-model", f"{PROJECT_ROOT}/models/{MODEL_NAME}", "--output",
    f"{PROJECT_ROOT}/models/onnx/{MODEL_NAME}.onnx"
]

process = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)

output, error = process.communicate()
print(output)
