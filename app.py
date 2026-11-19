from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from PIL import Image, UnidentifiedImageError
import io
import logging

from stanford_dogs.inference import ONNXDogClassifier
from stanford_dogs.config import ONNX_Model_Path
from stanford_dogs.utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"PNG", "JPEG"}

app = FastAPI()
model = ONNXDogClassifier(ONNX_Model_Path)

class PredictResponse(BaseModel):
    predicted_class: str

@app.get("/health", summary="Health Check", tags=["Status"])
async def health_check():
    return {"status": "ok", "message": "API is running"}

@app.post("/predict", summary="Predict Dog Breed", tags=["Inference"], response_model= PredictResponse)
async def predict(file: UploadFile = File(...)):

    contents = await file.read(MAX_FILE_SIZE + 1)
    
    if not contents:
        raise HTTPException(400, "Empty file.")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large.")

    # Validate actual image content.
    try:
        image = Image.open(io.BytesIO(contents))

        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(400, "Only PNG and JPEG are supported.")

        image.verify()
        image = Image.open(io.BytesIO(contents))
        image.load()
        
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(400, "Invalid or corrupted image.")

    
    predicted_class = await run_in_threadpool(model.predict, img_source=image)
    logger.info("Prediction success for %s: %s", file.filename, predicted_class)
    return {"predicted_class": predicted_class}