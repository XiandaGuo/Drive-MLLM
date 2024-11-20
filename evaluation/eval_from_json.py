import json
import ast
import re
import numpy as np
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from PIL import ImageDraw, ImageFont
import logging
from tqdm import tqdm
from typing import List, Tuple, Set
import argparse
import os
import math
from random import random

from prompt.prompt_idx import PromptIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box1, box2 (List[int]): [x_min, y_min, x_max, y_max].
    Returns:
        float: IoU score (0 to 1).
    """

    box1, box2 = np.array(box1), np.array(box2)
    xi1, yi1 = np.maximum(box1[:2], box2[:2]) # Intersection top-left
    xi2, yi2 = np.minimum(box1[2:], box2[2:]) # Intersection bottom-right 
    inter_width, inter_height = np.maximum(xi2 - xi1, 0), np.maximum(yi2 - yi1, 0)
    inter_area = inter_width * inter_height
    box1_area = np.prod(box1[2:] - box1[:2])  # Area of box1
    box2_area = np.prod(box2[2:] - box2[:2])  # Area of box2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def get_score(err: float, alpha: float) -> float:
    """Calculate score by alpha parameter."""
    return 1 / (1 + alpha * err)


def main(args):
    eval_root_dir = Path(args.eval_root_dir)
    eval_model_path = args.eval_model_path
    hf_dataset = args.hf_dataset
    save_dir = Path(args.save_dir)

    # Load dataset
    logger.info("Load dataset...")
    dataset = load_dataset(hf_dataset, split='validation')
    logger.info(f"Loaded {len(dataset)} data from the dataset.")

    # Get model paths
    model_paths = [eval_model_path] if eval_model_path.lower() != 'all' else [
        os.path.join(f, s) for f in os.listdir(eval_root_dir) for s in os.listdir(os.path.join(eval_root_dir, f))
    ]
   
    lengths_should_be = [880, 880, 191, 191, 191, 191, 880, 880, 191, 191]
    logger.info(f"Evaluating models: {model_paths}.")
    for model_path in model_paths:

        eval_dir = eval_root_dir / model_path
        shot_dirs = sorted(eval_dir.iterdir())
        for shot_dir in shot_dirs:
            logger.info(f"Processing model: {model_path}, shot: {shot_dir.name}.")
            inference_sample_index_file_name = "inference_sample_index.json"
    
            mllm_output_files_dir = shot_dir
            mllm_output_files = os.listdir(mllm_output_files_dir)
            mllm_output_files = [file for file in mllm_output_files if file.endswith('.json')]
            mllm_output_files.remove(inference_sample_index_file_name)
            mllm_output_files.sort()
            inference_sample_index_file = mllm_output_files_dir / inference_sample_index_file_name
            with open(inference_sample_index_file, 'r') as file:
                inference_sample_index = json.load(file)

            pattern = r"<OUTPUT FORMAT START>(.*?)<OUTPUT FORMAT END>"
            image_width, image_height = dataset[0]['image'].size
            eval_result = []
            for file_index in range(len(mllm_output_files)):
                mllm_output_file = mllm_output_files_dir / mllm_output_files[file_index]
                logger.info(f"Load MLLM output from {mllm_output_file}...")
                with open(mllm_output_file, 'r') as file:
                    mllm_output_data = json.load(file)

                match_mllm_output_list = []
                no_match_cnt = 0
                for data in mllm_output_data:
                    try:
                        match = re.search(pattern, data['output'], re.DOTALL)
                        if not match:
                            raise ValueError(f"No match found for pattern: {pattern}")
                        content = match.group(1).strip()
                        match_mllm_output_list.append(dict(dataset_index=data['dataset_index'], output=content))
                    except ValueError:
                        no_match_cnt += 1
                        continue

                wrong_cnt = 0
                correct_mllm_output_list = []

                if file_index == PromptIndex.XY_QA.value:
                    for data in match_mllm_output_list:
                        try:
                            xy = ast.literal_eval(data['output'].strip())
                            assert (0 <= xy[0] <= image_width-1) and (0 <= xy[1] <= image_height-1)
                            correct_mllm_output_list.append(dict(dataset_index=data['dataset_index'], output=xy))
                        except Exception as e:
                            wrong_cnt += 1

                    error_sum = 0
                    for data in correct_mllm_output_list:
                        dataset_index = data['dataset_index']
                        gt_index = inference_sample_index[file_index][dataset_index]['inference_sample_index'][0]
                        gt = dataset[dataset_index]['location2D'][gt_index]
                        xy = data['output']
                        error_sum += math.sqrt((xy[0]-gt[0])**2 + (xy[1]-gt[1])**2)
                    if len(correct_mllm_output_list)==0:
                        xy_score = 0.0
                    else:
                        mse = error_sum / len(correct_mllm_output_list)
                        xy_score = round(get_score(mse, 0.005) * 100, 2)
                    eval_result.append(dict(metric="xy_score", value=xy_score))

                elif file_index == PromptIndex.BOX_QA.value:
                    for data in match_mllm_output_list:
                        try:
                            box = ast.literal_eval(data['output'].strip())
                            if "gemini" in model_path:
                                ymin, xmin, ymax, xmax = box
                                box = [
                                    xmin/1000*image_width,
                                    ymin/1000*image_height,
                                    xmax/1000*image_width,
                                    ymax/1000*image_height
                                ]
                            
                            assert (0 <= box[0] <= image_width-1) and (0 <= box[1] <= image_height-1) and (0 <= box[2] <= image_width-1) and (0 <= box[3] <= image_height-1)
                            correct_mllm_output_list.append(dict(dataset_index=data['dataset_index'], output=box))
                        except Exception as e:
                            wrong_cnt += 1

                    error_sum = 0
                    for data in correct_mllm_output_list:
                        dataset_index = data['dataset_index']
                        gt_index = inference_sample_index[file_index][dataset_index]['inference_sample_index'][0]
                        gt = dataset[dataset_index]['bboxes2D'][gt_index]
                        box = data['output']
                        
                        error_sum += iou(gt, box)
                    if len(correct_mllm_output_list)==0:
                        box_score = 0.0
                    else:
                        box_score = round(error_sum / len(correct_mllm_output_list)*100,2)
                    eval_result.append(dict(metric="box_score", value=box_score))

                elif file_index in [PromptIndex.LEFT_QA.value, PromptIndex.RIGHT_QA.value]:
                    for data in match_mllm_output_list:
                        try:
                            left_right_object = data['output'].strip().replace('the ', '')
                            assert left_right_object in dataset[data['dataset_index']]['descriptions']
                            correct_mllm_output_list.append(dict(dataset_index=data['dataset_index'], output=left_right_object))
                        except Exception as e:
                            wrong_cnt += 1

                    correct_cnt = 0
                    for data in correct_mllm_output_list:
                        dataset_index = data['dataset_index'] 

                        gt_index = inference_sample_index[file_index][dataset_index]['inference_sample_index']
                        object1_xy = dataset[dataset_index]['location2D'][gt_index[0]]
                        object2_xy = dataset[dataset_index]['location2D'][gt_index[1]]
                        object1_description = dataset[dataset_index]['descriptions'][gt_index[0]]
                        object2_description = dataset[dataset_index]['descriptions'][gt_index[1]]
                        if file_index == PromptIndex.LEFT_QA.value:
                            gt = object1_description if object1_xy[0] < object2_xy[0] else object2_description
                        else:
                            gt = object1_description if object1_xy[0] > object2_xy[0] else object2_description
                        object_description = data['output']
                        if object_description == gt:
                            correct_cnt +=1
                    if len(correct_mllm_output_list)==0:
                        left_right_accuracy= 0.0
                    else:
                        left_right_accuracy = round(correct_cnt / len(correct_mllm_output_list)*100, 2)
                    if file_index == PromptIndex.LEFT_QA.value:
                        eval_result.append(dict(metric="left_score", value=left_right_accuracy))
                    else:
                        eval_result.append(dict(metric="right_score", value=left_right_accuracy))
                
                elif file_index in [PromptIndex.FRONT_QA.value, PromptIndex.BEHIND_QA.value]:
                    for data in match_mllm_output_list:
                        try:
                            is_front_behind = data['output'].strip()
                            assert is_front_behind in ['Yes', 'No']
                            correct_mllm_output_list.append(dict(dataset_index=data['dataset_index'], output=is_front_behind))
                        except Exception as e:
                            wrong_cnt += 1

                    correct_cnt = 0
                    for data in correct_mllm_output_list:
                        dataset_index = data['dataset_index'] 

                        gt_index = inference_sample_index[file_index][dataset_index]['inference_sample_index']
                        object1_xyz = dataset[dataset_index]['location3D'][gt_index[0]]
                        object2_xyz = dataset[dataset_index]['location3D'][gt_index[1]]
                        if file_index == PromptIndex.FRONT_QA.value:
                            gt = 'Yes' if object1_xyz[2] < object2_xyz[2] else 'No'
                        else:
                            gt = 'Yes' if object1_xyz[2] > object2_xyz[2] else 'No'
                        object_judge = data['output']
                        if object_judge == gt:
                            correct_cnt +=1
                    if len(correct_mllm_output_list)==0:
                        front_behind_accuracy = 0.0
                    else:
                        front_behind_accuracy = round(correct_cnt / len(correct_mllm_output_list)*100,2)
                    
                    if file_index == PromptIndex.FRONT_QA.value:
                        eval_result.append(dict(metric="front_score", value=front_behind_accuracy))
                    else:
                        eval_result.append(dict(metric="behind_score", value=front_behind_accuracy))

                elif file_index in [PromptIndex.DA_QA.value, PromptIndex.DZ_QA.value, PromptIndex.DAB_QA.value, PromptIndex.DX_QA.value]:
                    for data in match_mllm_output_list:
                        try:
                            distance = float(data['output'].strip())
                            correct_mllm_output_list.append(dict(dataset_index=data['dataset_index'], output=distance))
                        except Exception as e:
                            wrong_cnt += 1
                    #
                    error_sum = 0
                    for data in correct_mllm_output_list:
                        dataset_index = data['dataset_index'] 
                        gt_index = inference_sample_index[file_index][dataset_index]['inference_sample_index']
                        if file_index == PromptIndex.DA_QA.value:
                            object_xyz = dataset[dataset_index]['location3D'][gt_index[0]]
                            gt = math.sqrt(object_xyz[0]**2+object_xyz[1]**2+object_xyz[2]**2)
                        elif file_index == PromptIndex.DZ_QA.value:
                            object_xyz = dataset[dataset_index]['location3D'][gt_index[0]]
                            gt = object_xyz[2]
                        elif file_index == PromptIndex.DAB_QA.value: 
                            object1_xyz = dataset[dataset_index]['location3D'][gt_index[0]]
                            object2_xyz = dataset[dataset_index]['location3D'][gt_index[1]]
                            gt = math.sqrt((object1_xyz[0]-object2_xyz[0])**2+(object1_xyz[1]-object2_xyz[1])**2+(object1_xyz[2]-object2_xyz[2])**2)
                        else:
                            object1_xyz = dataset[dataset_index]['location3D'][gt_index[0]]
                            object2_xyz = dataset[dataset_index]['location3D'][gt_index[1]]
                            gt = abs(object1_xyz[0]-object2_xyz[0])

                        distance = data['output']
                        error_sum += abs(distance - gt)
                    if len(correct_mllm_output_list)==0:
                        distance_score= 0.0
                    else:
                        m_abs_err = error_sum / len(correct_mllm_output_list)
                        distance_score = round(get_score(m_abs_err, 0.05)*100,2)

                    if file_index == PromptIndex.DA_QA.value:
                        eval_result.append(dict(metric="dA_score", value=distance_score))
                    elif file_index == PromptIndex.DZ_QA.value:
                        eval_result.append(dict(metric="dz_score", value=distance_score))
                    elif file_index == PromptIndex.DAB_QA.value: 
                        eval_result.append(dict(metric="dAB_score", value=distance_score))
                    else:
                        eval_result.append(dict(metric="dx_score", value=distance_score))
                    
                eval_result.append(dict(metric=f"{file_index}_effective_ratio_str", value=f"{len(correct_mllm_output_list)}/{len(match_mllm_output_list)}/{len(mllm_output_data)}/{lengths_should_be[file_index]}"))
                eval_result.append(dict(metric=f"{file_index}_effective_ratio", value=len(correct_mllm_output_list)/lengths_should_be[file_index]*100))
                logger.info(f"Correct/Match/Total/Should data: {len(correct_mllm_output_list)}/{len(match_mllm_output_list)}/{len(mllm_output_data)}/{lengths_should_be[file_index]}")     
            
            # save
            save_path  = save_dir / model_path / shot_dir.name
            save_path.mkdir(exist_ok=True, parents =True)
            df = pd.DataFrame(eval_result)
            df.to_csv(save_path / f'eval_result.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MLLM output.")
    parser.add_argument('--hf_dataset', type=str, default='bonbon-rj/MLLM_eval_dataset',
                        help='Specify the path to the Hugging Face dataset to use for evaluation.')
    parser.add_argument('--eval_root_dir', type=str, default='inference/mllm_outputs',
                        help='Specify the root directory for MLLM output files.')
    parser.add_argument('--eval_model_path', type=str, default='all',
                        help='Specify the path of the model to evaluate.')
    parser.add_argument('--save_dir', type=str, default='evaluation/eval_result',
                        help='Specify the directory where evaluation results will be saved.')
    args = parser.parse_args()
    main(args)
