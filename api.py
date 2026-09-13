"""FastAPI REST Service for Model Deployment and Inference."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from validator import SinglePredictionRequest, BatchPredictionRequest, PredictionResponse
from inference import ModelPredictor
from logger import logger

app = FastAPI(
    title="ML-Project-01 REST API",
    description="High-performance machine learning inference microservice.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loaded predictor
predictor = None


@app.on_event("startup")
def load_model_on_startup():
    global predictor
    try:
        predictor = ModelPredictor()
        logger.info("FastAPI: Model predictor loaded successfully.")
    except Exception as e:
        logger.warning(f"FastAPI: Model predictor not loaded on startup: {e}")


@app.get("/", tags=["Health"])
def root():
    return {"message": "ML-Project-01 API is running", "status": "online"}


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(payload: SinglePredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model predictor is not initialized.")

    try:
        val = predictor.predict_single(payload.model_dump())
        return PredictionResponse(prediction=val, status="success")
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
