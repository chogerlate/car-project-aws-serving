import numpy as np
import pandas as pd
import copy
import supervision as sv
import torch
import json
import cv2

def pre_processing_results(yolo_results):
    results = []
    id2class = yolo_results[0].names
    for result in yolo_results:
        result = result.cpu().numpy()
        data_dict = dict()
        data_dict['orig_img'] = result.orig_img
        data_dict['path'] = result.path
        data_dict["orig_shape"] = result.orig_shape
        data_dict["class_ids"] = result.boxes.cls
        data_dict['classes'] = [id2class[i] for i in result.boxes.cls] 
        data_dict["boxes"] = result.boxes.xyxyn
        data_dict['conf'] = result.boxes.conf
        data_dict["instance_ids"] = [i for i in range(len(result.boxes.cls))]
        
         # "masks" error handling
        if result.masks is None:
            data_dict["masks"] = np.array([[[]]])
        else :
            data_dict["masks"] = result.masks.data 
            
        results.append(data_dict)
         
    return results    

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def detection_result_to_json_format(result: dict):
    """convert pre processed prediction result to json format

    Args:
        result (dict): _description_
    """
    prediction_list = []
    for id in range(len(result['classes'])):
        result_dict = dict()
        # extract bounding box to xyxyn format
        box = result['boxes'][id]
        result_dict['x1'] = box[0]
        result_dict['y1'] = box[1]
        result_dict['x2'] = box[2]
        result_dict['y2'] = box[3]
        # confience score
        result_dict['confidence'] = result['conf'][id]
        # class
        result_dict['class'] = result['classes'][id]
        # coordinate points of polygons
        polygons = sv.mask_to_polygons(result['masks'][id])
        # get the first polygon
        points = [{'x': float(row[0]), 'y': float(row[1])} for row in polygons[0]]
        result_dict['points'] = points
        # get class ID
        result_dict['class_id'] = result['class_ids'][id] 
        # severity if any
        if 'severity_id' in result:
            result_dict['severity_id'] = result['severity_id'][id]
            result_dict['severity'] = result['severity'][id]
        # instance order
        result_dict['instance_id'] = result['instance_ids'][id]
        
        # add to list
        prediction_list.append(result_dict)
    return prediction_list

def find_intercept_report(car_damage_results, car_part_results):
    interception_instances = {
        "car_part": [],
        "damage": [], 
        # "damage id":[],
    }
    for part_id, part_mask in enumerate(car_part_results['masks']):
        
        interception_damages = []
        interception_ids = [] 
        for damage_id, damage_mask in enumerate(car_damage_results['masks']):
            try :
                if np.sum(damage_mask * part_mask) > 20:
                    if car_damage_results['classes'][damage_id] == "deformation":
                        damage_name = car_damage_results["severity"][damage_id]+ " " + car_damage_results['classes'][damage_id]
                    else :
                        damage_name = car_damage_results['classes'][damage_id]
                    # interception_ids.append(damage_id)
                    interception_damages.append(damage_name)
            except:
                pass    
                
        # add row of interception
        # interception_instances['damage id'].append(interception_ids)
        interception_instances['car_part'].append(car_part_results['classes'][part_id])
        # convert list of damage to string
        interception_damages = ', '.join(interception_damages) if len(interception_damages) > 0 else ""
        interception_instances['damage'].append(interception_damages)
    report = pd.DataFrame(interception_instances)
    
    return report


def export_result_to_json(  inputs, car_part_model_results, car_damage_model_results, car_part_model, car_damage_model):
    pipeline_result = dict()
    for image_id in range(len(inputs)):
        image_result = dict()
        # add image meta data 
        image_result["image_meta_data"] = { 
            "orig_shape": car_part_model_results[image_id]['orig_shape'],
            "n_car_parts": len(car_part_model_results[image_id]['classes']),
            "n_car_damages": len(car_damage_model_results[image_id]['classes']) 
        }
        image_result['car_part_results'] = detection_result_to_json_format(car_part_model_results[image_id])
        image_result['car_damage_results'] = detection_result_to_json_format(car_damage_model_results[image_id])
        image_result['report'] = find_intercept_report(car_damage_model_results[image_id],car_part_model_results[image_id]).to_dict(orient="list")
        
        # add to json result
        pipeline_result[inputs[image_id]] = image_result 

    
    # add meta data
    pipeline_result["meta_data"] = {
        "car_part_id_to_class":  car_part_model.names,
        "car_damage_id_to_class": car_damage_model.names,
        "damage_severity_id_to_class": {
            0: "None",
            1: "minor",
            2: "moderate",
            3: "severe"
            }
    }
    return pipeline_result


def find_intercept_masks(damage_results, car_part_results):
    intercept_instances = {
        "car_part_inst_ids": [],
        "damage_inst_ids": [],
    }
    for part_id, part_mask in enumerate(car_part_results['masks']):
        damage_instance = []
        for damage_id, damage_mask in enumerate(damage_results['masks']):
            if np.sum(damage_mask * part_mask) > 20:
                damage_instance.append(damage_results['instance_ids'][damage_id])
        intercept_instances['car_part_inst_ids'].append(car_part_results['instance_ids'][part_id])
        intercept_instances['damage_inst_ids'].append(damage_instance)
    return intercept_instances

def severity_determiner(model, car_damage_model_result):
    device = model.device
    result = copy.deepcopy(car_damage_model_result)
    severity = []
    for id in  range(len(result['classes'])):
        cat = result['classes'][id]
        if cat == "deformation" :
            # load box that was in xyxyn format
            box = result['boxes'][id]
            # rescale transform them to xyxy format
            orig_shape = result['orig_shape']
            box[0],box[2] = box[0]*orig_shape[1], box[2]*orig_shape[1]
            box[1],box[3] = box[1]*orig_shape[0], box[3]*orig_shape[0]
            
            # load numpy image and crop image
            cropped_image = sv.crop_image(image=result['orig_img'], xyxy=box)
            # resize image to 224x224 before feeding into model
            resized_image = cv2.resize(cropped_image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            # convert image to tensor
            x = torch.from_numpy(resized_image)
            x = x.permute(2, 0, 1) # from NHWC to NCHW
            x = x.float() # convert to float
            x = x.unsqueeze(0) # add batch dimension
            x = x.to(device) # set device to cuda if available 
            with torch.no_grad():
                model.eval() # set model to evaluation mode
                pred = model(x).cpu().item() # result to cpu and get value instead of tensor
            severity.append(pred+1)
        else :
            severity.append(0)
        
    return severity