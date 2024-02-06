# car-project-aws-serving
Codebase for deploying car damage severity detection model on the cloud.

## Tech Stack
**AI**: pytorch, pytorch lightning, ultralytics

**Deploy**: FastAPI, Docker, AWS ECS, ECR

## .env setup 
```
# AWS  

# S3
S3_BUCKET = "BUCKET_NAME"
S3_DEFORMATION_OBJECT_NAME = "DEFORMATION_CLASSIFICATION.ckpt"
S3_CAR_PART_OBJECT_NAME = "CAR_PART_DETECTION.pt"
S3_CAR_DAMAGE_OBJECT_NAME = "CAR_DAMAGE_DETECTION.pt"

# model path
CAR_PART_PATH = "./CAR_PART_DETECTION.pt"
CAR_DAMAGE_PATH = "./CAR_DAMAGE_DETECTION.pt"
DEFORMATION_PATH = "./DEFORMATION_CLASSIFICATION.ckpt"
```


## Local test
```
    cd ./app 
    python3 main.py
```

## Docker build
```
docker build -t car-project .
```

## Docker run 
You need to specify your AWS credentials in Docker env
```
docker run -p 8080:80 -d car-project
```

## Prediction API request format

```
{
  "urls": [
    "string"
  ],
  "car_part_conf_thres": float | 0.3,
  "car_part_iou_thres": float | 0.5,
  "car_damage_conf_thres": float | 0.3,
  "car_damage_iou_thres": float | 0.5
}
```

## API Prediction Result format ( Example )

Noting :
- The bounding box is in xyxyn format.
- I made this easy for humans to understand, so it is not runable in any program.
  
```
{
    "url":{
        "image_meta_data":{
            "origin_shape": [float,float],
            "n_car_parts": int,
            "n_car_damages": int, 
        },
        "car_part_results":[
            {
                x1: float,
                y1: float,
                x2: float,
                y2: float,
                confidence: float,
                class: str,
                points: LIST[DICT],
                class_id: int,
                instance_id: int

            },
            {
                x1: float,
                y1: float,
                x2: float,
                y2: float,
                confidence: float,
                class: str,
                points: LIST[DICT],
                class_id: int,
                instance_id: int
            }
        ],
        "car_damage_results":[
            {
                x1: float,
                y1: float,
                x2: float,
                y2: float,
                confidence: float,
                class: str,
                points: LIST[DICT],
                class_id: int,
                instance_id: int
                # new part
                severity_id: int
                severity: str,
            }
        ],
        "report":{
            "car_part":[],
            "damage":[]
        }
    },
    "meta_data":{
        "car_part_id_to_class": DICT,
        "car_damage_id_to_class": DICT,
        "damage_severity_id_to_class": DICT
    }


}
    
```
