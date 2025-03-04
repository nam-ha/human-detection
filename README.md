# Human Detection API
![image](https://github.com/user-attachments/assets/e1d81cb1-4bcf-4938-85ea-662245afeb82)

This project implements a human detection system using YOLOv8 as the backbone model, with an API developed using FastAPI.

## Getting Started
1. Environment Configuration \
Create a .env file in the project root based on the .env.example file.

2. Deploy using Docker
```sh
  docker-compose up --build
```
This will:
- Connect to the PostgreSQL database
- Start the FastAPI server

## **API Endpoints**

### **Upload Image and Detect People**
- **Endpoint**: `POST /api/v1/predict`
- **Request**:
  - `b64image`: `str` (base64 encoded image)
  - `confidence_threshold`: `float` (ranging from 0.0 to 1.0)
- **Response**:
  - `b64image`: `str` (base64 visualized image)
  - `num_humans`: `int` (number of humans detected)

## **Database Schema**
The following data is stored for each detection:

| Column             | Type | Description                             |
|--------------------|------|-----------------------------------------|
| `query_id`         | str  | Unique ID                               |
| `time`             | str  | Time the query was received             |
| `query_image_file` | str  | Input file path in media storage        |
| `result_image_file`| str  | Result file path in media storage       |
| `num_humans`       | int  | Number of detected humans in query image|

## **Tech Stack**
- **Backend**: Python, FastAPI (with Pydantic for data validation)
- **Detection Model**: YOLOv8
- **Database**: PostgreSQL + SQLAlchemy
- **Frontend**: Next.js (not included in this repo)
- **Deployment**: Docker Compose
