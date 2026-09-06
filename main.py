from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np

from PIL import Image
import io
import json
import os


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "model/fruit_quality_saved_model"
CLASS_PATH = "model/class_names.json"

IMG_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.70


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Fruit Quality Classification API",
    description="Predicts fruit/product category and quality from an image",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CHECK FILES
# =========================================================

print("Checking model files...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"SavedModel not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError(
        f"Class file not found: {CLASS_PATH}"
    )


# =========================================================
# LOAD SAVED MODEL
# =========================================================

print("Loading SavedModel...")

try:

    model = tf.keras.layers.TFSMLayer(
        MODEL_PATH,
        call_endpoint="serve"
    )

    print("✅ SavedModel loaded successfully")

except Exception as e:

    print("❌ Model loading failed")

    raise RuntimeError(
        f"Could not load SavedModel: {str(e)}"
    )


# =========================================================
# LOAD CLASS INFORMATION
# =========================================================

with open(CLASS_PATH, "r") as f:

    class_info = json.load(f)


product_names = class_info["products"]
quality_names = class_info["qualities"]


print("Products:", product_names)
print("Qualities:", quality_names)


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Fruit Quality Classification API",
        "status": "running",
        "model": "SavedModel + TFSMLayer",
        "endpoint": "/predict"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_bytes):

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = image.convert("RGB")

        image = image.resize(
            IMG_SIZE
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {str(e)}"
        )


# =========================================================
# GET MODEL OUTPUT
# =========================================================

def get_predictions(image):

    try:

        # TFSMLayer is called directly.
        # DO NOT use model.predict()

        outputs = model(image)

        print("Raw model output type:", type(outputs))

        # -------------------------------------------------
        # Case 1: Dictionary output
        # -------------------------------------------------

        if isinstance(outputs, dict):

            print("Output keys:", outputs.keys())

            if "product_output" in outputs:
                product_prediction = outputs["product_output"]

            elif "product" in outputs:
                product_prediction = outputs["product"]

            else:
                raise RuntimeError(
                    f"Product output not found. Available keys: {list(outputs.keys())}"
                )

            if "quality_output" in outputs:
                quality_prediction = outputs["quality_output"]

            elif "quality" in outputs:
                quality_prediction = outputs["quality"]

            else:
                raise RuntimeError(
                    f"Quality output not found. Available keys: {list(outputs.keys())}"
                )

        # -------------------------------------------------
        # Case 2: List / Tuple output
        # -------------------------------------------------

        elif isinstance(outputs, (list, tuple)):

            if len(outputs) != 2:

                raise RuntimeError(
                    f"Expected 2 outputs but received {len(outputs)}"
                )

            product_prediction = outputs[0]
            quality_prediction = outputs[1]

        else:

            raise RuntimeError(
                f"Unknown model output type: {type(outputs)}"
            )


        # Convert TensorFlow tensors to NumPy

        product_prediction = product_prediction.numpy()

        quality_prediction = quality_prediction.numpy()


        return product_prediction, quality_prediction


    except Exception as e:

        print("Prediction error:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(e)}"
        )


# =========================================================
# PREDICT ENDPOINT
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Check file type
    # -----------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG, PNG or WEBP image."
        )


    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    image_bytes = await file.read()


    if len(image_bytes) == 0:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )


    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    image = preprocess_image(
        image_bytes
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    product_prediction, quality_prediction = get_predictions(
        image
    )


    # -----------------------------------------------------
    # PRODUCT
    # -----------------------------------------------------

    product_id = int(
        np.argmax(product_prediction[0])
    )


    product_confidence = float(
        product_prediction[0][product_id]
    )


    if product_id >= len(product_names):

        raise HTTPException(
            status_code=500,
            detail="Product class index does not match class_names.json"
        )


    product = product_names[
        product_id
    ]


    # -----------------------------------------------------
    # QUALITY
    # -----------------------------------------------------

    quality_id = int(
        np.argmax(quality_prediction[0])
    )


    quality_confidence = float(
        quality_prediction[0][quality_id]
    )


    if quality_id >= len(quality_names):

        raise HTTPException(
            status_code=500,
            detail="Quality class index does not match class_names.json"
        )


    quality = quality_names[
        quality_id
    ]


    # -----------------------------------------------------
    # CONFIDENCE CHECK
    # -----------------------------------------------------

    accepted = (
        product_confidence >= CONFIDENCE_THRESHOLD
        and
        quality_confidence >= CONFIDENCE_THRESHOLD
    )


    # -----------------------------------------------------
    # LOW CONFIDENCE RESPONSE
    # -----------------------------------------------------

    if not accepted:

        return {

            "success": False,

            "message": (
                "Low confidence prediction. "
                "Please upload a clearer fruit image."
            ),

            "filename": file.filename,

            "product": product,

            "product_confidence": round(
                product_confidence * 100,
                2
            ),

            "quality": quality,

            "quality_confidence": round(
                quality_confidence * 100,
                2
            )
        }


    # -----------------------------------------------------
    # SUCCESS RESPONSE
    # -----------------------------------------------------

    return {

        "success": True,

        "message": "Prediction successful",

        "filename": file.filename,

        "product": product,

        "product_confidence": round(
            product_confidence * 100,
            2
        ),

        "quality": quality,

        "quality_confidence": round(
            quality_confidence * 100,
            2
        )
    }