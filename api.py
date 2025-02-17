import os

import uvicorn

from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from source.core import HumanDetector
from source.utils.image import BBoxDrawer

from configs.general import env_config, paths_config

class PredictRequest(BaseModel):
    b64image: str
    confidence_threshold: float

class PredictResponse(BaseModel):
    b64image: str
    num_humans: int

# ==

detector = HumanDetector()

detector.load_model(
    model_file = os.path.join(
        paths_config.models_folder,
        'finetuned',
        'human-1.pt'
    )
)

bbox_drawer = BBoxDrawer()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

router = APIRouter(prefix = "/api/v1")

@router.post("/predict")
async def predict(request: PredictRequest):            
    predictions = detector.predict_b64image(
        b64image = request.b64image,
        confidence_threshold = request.confidence_threshold
    )
    
    prediction = predictions[0]
    
    xywhs = prediction.boxes.xywh.tolist()
    classes = prediction.boxes.cls.tolist()
    confidences = prediction.boxes.conf.tolist()
    
    num_detected_objects = len(xywhs)
        
    drawn_b64image = bbox_drawer.draw_on_b64image(
        b64image = request.b64image,
        xywhs = xywhs,
        labels = ['human' for _ in range(len(xywhs))],
        colors = ['red' for _ in range(len(xywhs))],
        line_width = 2
    )
    
    return PredictResponse(
        b64image = drawn_b64image,
        num_humans = num_detected_objects
    )

app.include_router(router)

def main(): 
    uvicorn.run(
        app, 
        host = "0.0.0.0", 
        port = env_config.port
    )
    
if __name__ == '__main__':
    main()
