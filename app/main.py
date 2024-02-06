from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.model import cnnModel
from ultralytics import YOLO
from app.utils import *
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import Response
import torch
import json
from dotenv import load_dotenv
import os 
import uvicorn
import boto3 


load_dotenv()


S3_BUCKET = os.getenv('S3_BUCKET')
S3_CAR_PART_OBJECT_NAME = os.getenv('S3_CAR_PART_OBJECT_NAME')
S3_CAR_DAMAGE_OBJECT_NAME = os.getenv('S3_CAR_DAMAGE_OBJECT_NAME')
S3_DEFORMATION_OBJECT_NAME = os.getenv('S3_DEFORMATION_OBJECT_NAME')

CAR_PART_PATH = "./car-part-detection-weight.pt"
CAR_DAMAGE_PATH = "./car-damage-detection-weight.pt"
DEFORMATION_PATH = "./deformation_classification.ckpt"

# print("load env: ",type(S3_BUCKET)) # sanity check

s3 = boto3.client('s3', region_name='ap-southeast-1')
s3.download_file(S3_BUCKET, S3_CAR_PART_OBJECT_NAME, CAR_PART_PATH)
s3.download_file(S3_BUCKET, S3_CAR_DAMAGE_OBJECT_NAME, CAR_DAMAGE_PATH) 
s3.download_file(S3_BUCKET, S3_DEFORMATION_OBJECT_NAME, DEFORMATION_PATH)


app = FastAPI()

# identify device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' 
# Load model and set device
car_part_model = YOLO(CAR_PART_PATH).to(device)
car_damage_model = YOLO(CAR_DAMAGE_PATH).to(device)
deformation_model = cnnModel.load_from_checkpoint(DEFORMATION_PATH).to(device)

# convertor for converting ids to classes
damage_severity_id_to_class = {
            0: "None",
            1: "minor",
            2: "moderate",
            3: "severe"
            }

class image_payload(BaseModel):
    urls: list[str]
    car_part_conf_thres: float = 0.3
    car_part_iou_thres: float = 0.5
    car_damage_conf_thres: float = 0.3
    car_damage_iou_thres: float = 0.5

@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.post("/predict")
def prediction(payload: image_payload) :
    car_part_results = car_part_model(payload.urls, imgsz = 640 , conf=payload.car_part_conf_thres, iou=payload.car_part_iou_thres)
    car_damage_results = car_damage_model(payload.urls, imgsz = 640,conf=payload.car_damage_conf_thres, iou=payload.car_damage_iou_thres)
    
    car_part_results = pre_processing_results(car_part_results)
    car_damage_results = pre_processing_results(car_damage_results)
    
    for i in range(len(car_damage_results)):
        severity = severity_determiner(deformation_model, car_damage_results[i])
        car_damage_results[i]['severity_id'] = severity
        car_damage_results[i]['severity'] = [damage_severity_id_to_class[i] for i in severity] 
    
    pipeline_result = export_result_to_json(payload.urls, car_part_results, car_damage_results, car_part_model, car_damage_model)
    json_result = json.dumps(pipeline_result, cls =NumpyArrayEncoder)

    return Response(json_result, media_type='application/json',status_code=200)


if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8080)
