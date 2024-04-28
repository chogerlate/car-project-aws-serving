from typing import Union
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel
from ultralytics import YOLO
from app.utils import *
from app.model import cnnModel
# from utils import *
# from model import cnnModel
from requests.exceptions import RequestException, MissingSchema
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from PIL import Image, UnidentifiedImageError
from fastapi import Response
import torch
import json
from dotenv import load_dotenv
import os 
import uvicorn
import boto3 
import urllib.request
import requests

load_dotenv()


S3_BUCKET = os.getenv('S3_BUCKET')
S3_CAR_PART_OBJECT_NAME = os.getenv('S3_CAR_PART_OBJECT_NAME')
S3_CAR_DAMAGE_OBJECT_NAME = os.getenv('S3_CAR_DAMAGE_OBJECT_NAME')
S3_DEFORMATION_OBJECT_NAME = os.getenv('S3_DEFORMATION_OBJECT_NAME')

CAR_PART_PATH = "./car-part-detection-weight.pt"
CAR_DAMAGE_PATH = "./car-damage-detection-weight.pt"
DEFORMATION_PATH = "./deformation_classification.ckpt"

# print("load env: ",type(S3_BUCKET)) # sanity check

# download model from s3
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


def download_and_open_image(url):
  """Downloads the image from the URL and opens it using PIL.

  Args:
      url: The URL of the image.

  Returns:
      A PIL Image object if successful, None otherwise.
  """
  try:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
      return Image.open(response.raw)
    else:
      return None
  except Exception as e:
    print(f"Error downloading image from {url}: {e}")
    return None
     

@app.post("/predict")
def prediction(payload: image_payload) :
    # initialize results
    car_part_results =  [] 
    car_damage_results = []

    for url in payload.urls:

        # download image from url 
        image = download_and_open_image(url)
        if not image: # if image is not downloaded
            return Response(json.dumps({"error": f"Failed to download image from url: {url}"}), media_type='application/json', status_code=400)

        car_part_results.extend(car_part_model.predict(image, imgsz=640, conf=payload.car_part_conf_thres, iou=payload.car_part_iou_thres, stream=True))
        car_damage_results.extend(car_damage_model.predict(image, imgsz=640, conf=payload.car_damage_conf_thres, iou=payload.car_damage_iou_thres, stream=True))


    car_part_results = pre_processing_results(car_part_results)
    car_damage_results = pre_processing_results(car_damage_results)
    # print(car_part_results)
    # print(car_damage_results)
    
    for i in range(len(car_damage_results)):
        try :
            severity = severity_determiner(deformation_model, car_damage_results[i])
            car_damage_results[i]['severity_id'] = severity
            car_damage_results[i]['severity'] = [damage_severity_id_to_class[i] for i in severity] 
        except:
            car_damage_results[i]['severity_id'] = [0 for i in range(len(car_damage_results[i]['classes']))]
            car_damage_results[i]['severity'] = ["None" for i in range(len(car_damage_results[i]['classes']))]

    pipeline_result = export_result_to_json(payload.urls, car_part_results, car_damage_results, car_part_model, car_damage_model)
    json_result = json.dumps(pipeline_result, cls =NumpyArrayEncoder)

    return Response(json_result, media_type='application/json',status_code=200)


if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8080)
