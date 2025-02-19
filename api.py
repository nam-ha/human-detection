import os

import uvicorn
import psycopg2
import base64

from fastapi import FastAPI, APIRouter

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field, field_validator

from io import BytesIO
from PIL import Image
from datetime import datetime

from source.core import HumanDetector
from source.modules.database import HumanDetectorDatabase, PredictionRecord
from source.utils.image import BBoxDrawer, save_b64image, strip_mime_prefix

from configs.general import env_config, paths_config

class PredictRequest(BaseModel):
    b64image: str
    confidence_threshold: float = Field(ge = 0.0, le = 1.0)

    @field_validator("b64image")
    @classmethod
    def validate_b64image(cls, b64image):
        b64image = strip_mime_prefix(b64image)
    
        try:
            image_data = base64.b64decode(b64image)
            
            image_buffer = BytesIO(image_data)
            
            pilimage = Image.open(image_buffer)
            
            pilimage.verify()
            
        except Exception:
            raise ValueError("Invalid base64 image data.")
        
        pilimage_size = pilimage.size[0] * pilimage.size[1]
        
        if pilimage_size < env_config.min_image_size:
            raise ValueError("Image too small.")
        
        if pilimage_size > env_config.max_image_size:
            raise ValueError("Image too large.")
        
        format = pilimage.format
        if format not in ['PNG', 'JPEG', 'JPG']:
            raise ValueError(f"Not supported image format: {format}")
            
        return b64image
        
class PredictResponse(BaseModel):
    b64image: str
    num_humans: int

def setup_folders():
    os.makedirs(
        paths_config.media_storage_folder,
        exist_ok = True
    )

def setup_database():
    conn = psycopg2.connect(
        dbname = env_config.database_name, 
        user = env_config.database_user,
        password = env_config.database_password, 
        host = env_config.database_host, 
        port = env_config.database_port
    )
    
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Predictions (
        query_id SERIAL PRIMARY KEY,
        time VARCHAR(32) NOT NULL,
        query_image_file VARCHAR(64) NOT NULL,
        result_image_file VARCHAR(64) NOT NULL,
        num_humans INT NOT NULL
    );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
        
# ==

database = HumanDetectorDatabase(
    database_url = env_config.database_url
)

detector = HumanDetector()

detector.load_model(
    model_file = os.path.join(
        paths_config.models_folder,
        'finetuned',
        env_config.model_filename
    )
)

bbox_drawer = BBoxDrawer()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

router = APIRouter(prefix = "/api/v1")

@router.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
    query_image_file = os.path.join(
        paths_config.media_storage_folder,
        'queries',
        f'{current_time}.png'
    )
    
    result_image_file = os.path.join(
        paths_config.media_storage_folder,
        'results',
        f'{current_time}.png'
    )
        
    predictions = detector.predict_b64image(
        b64image = request.b64image,
        confidence_threshold = request.confidence_threshold
    )
    
    prediction = predictions[0]
    
    xywhs = prediction.boxes.xywh.tolist()
    classes = prediction.boxes.cls.tolist()
    confidences = prediction.boxes.conf.tolist()
    
    num_detected_objects = len(xywhs)
        
    drawn_b64image = bbox_drawer.draw_bboxes_on_b64image(
        b64image = request.b64image,
        xywhs = xywhs,
        labels = ['human' for _ in range(len(xywhs))],
        confidences = confidences,
        colors = ['red' for _ in range(len(xywhs))],
        line_width = 2
    )
    
    save_b64image(
        b64image = request.b64image,
        save_file = query_image_file
    )
    
    save_b64image(
        b64image = drawn_b64image, 
        save_file = result_image_file
    )
    
    prediction_record = PredictionRecord(
        time = current_time,
        query_image_file = query_image_file,
        result_image_file = result_image_file,
        num_humans = num_detected_objects
    )
    
    database.add_record(prediction_record)
    
    return PredictResponse(
        b64image = drawn_b64image,
        num_humans = num_detected_objects
    )

app.include_router(router)

def main(): 
    setup_folders()
    setup_database()
    
    uvicorn.run(
        app, 
        host = "0.0.0.0", 
        port = env_config.port
    )
    
if __name__ == '__main__':
    main()
