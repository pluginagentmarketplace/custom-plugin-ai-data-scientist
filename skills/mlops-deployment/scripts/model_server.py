#!/usr/bin/env python3
"""
FastAPI Model Serving Server
Production-ready ML model deployment
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import time
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response Models
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Input features")
    request_id: Optional[str] = Field(None, description="Optional request ID")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [1.0, 2.0, 3.0, 4.0],
                "request_id": "req-001"
            }
        }


class BatchPredictionRequest(BaseModel):
    instances: List[List[float]] = Field(..., description="Batch of instances")
    request_id: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[float] = None
    request_id: Optional[str] = None
    latency_ms: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    probabilities: Optional[List[float]] = None
    request_id: Optional[str] = None
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_predictions: int
    avg_latency_ms: float
    error_rate: float
    model_version: str


# Global state
class ModelState:
    def __init__(self):
        self.model = None
        self.model_version = "unknown"
        self.start_time = time.time()
        self.prediction_count = 0
        self.error_count = 0
        self.total_latency = 0.0


state = ModelState()


def load_model(model_path: str = "model.pkl"):
    """Load model from file."""
    try:
        state.model = joblib.load(model_path)
        state.model_version = os.environ.get("MODEL_VERSION", "1.0.0")
        logger.info(f"Model loaded: version {state.model_version}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup/shutdown."""
    # Startup
    model_path = os.environ.get("MODEL_PATH", "model.pkl")
    if os.path.exists(model_path):
        load_model(model_path)
    else:
        logger.warning(f"Model file not found: {model_path}")
        # Create a dummy model for demo
        from sklearn.ensemble import RandomForestClassifier
        state.model = RandomForestClassifier()
        state.model.fit([[0, 0], [1, 1]], [0, 1])
        state.model_version = "demo-1.0.0"
        logger.info("Using demo model")

    yield

    # Shutdown
    logger.info("Shutting down model server")


# Create FastAPI app
app = FastAPI(
    title="ML Model Server",
    description="Production-ready ML model serving API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    start_time = time.time()

    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = float(state.model.predict(features)[0])

        # Get probability if available
        probability = None
        if hasattr(state.model, 'predict_proba'):
            proba = state.model.predict_proba(features)[0]
            probability = float(max(proba))

        latency = (time.time() - start_time) * 1000

        # Update metrics
        state.prediction_count += 1
        state.total_latency += latency

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            request_id=request.request_id,
            latency_ms=round(latency, 2),
            model_version=state.model_version
        )

    except Exception as e:
        state.error_count += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions."""
    start_time = time.time()

    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = np.array(request.instances)
        predictions = state.model.predict(features).tolist()

        probabilities = None
        if hasattr(state.model, 'predict_proba'):
            proba = state.model.predict_proba(features)
            probabilities = [float(max(p)) for p in proba]

        latency = (time.time() - start_time) * 1000

        state.prediction_count += len(predictions)
        state.total_latency += latency

        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            probabilities=probabilities,
            request_id=request.request_id,
            latency_ms=round(latency, 2),
            model_version=state.model_version
        )

    except Exception as e:
        state.error_count += 1
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if state.model is not None else "unhealthy",
        model_loaded=state.model is not None,
        model_version=state.model_version,
        uptime_seconds=round(time.time() - state.start_time, 2)
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Get server metrics."""
    total = state.prediction_count + state.error_count
    error_rate = state.error_count / total if total > 0 else 0.0
    avg_latency = state.total_latency / state.prediction_count if state.prediction_count > 0 else 0.0

    return MetricsResponse(
        total_predictions=state.prediction_count,
        avg_latency_ms=round(avg_latency, 2),
        error_rate=round(error_rate, 4),
        model_version=state.model_version
    )


@app.post("/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload model from disk."""
    model_path = os.environ.get("MODEL_PATH", "model.pkl")
    background_tasks.add_task(load_model, model_path)
    return {"status": "reload_initiated"}


def main():
    """Run the server."""
    import uvicorn
    uvicorn.run(
        "model_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4
    )


if __name__ == "__main__":
    main()
