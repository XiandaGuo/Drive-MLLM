import json
import ast
import re
from pathlib import Path
import pandas as pd
import logging
from typing import List, Tuple, Set
import argparse
import os
import math
import string

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_centerness(point, bbox):
    """
    Computes the centerness score of a point with respect to a bounding box.
    The closer the point is to the center of the box, the higher the score.
    """
    x, y = point
    xmin, ymin, xmax, ymax = bbox

    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return 0.0

    left = x - xmin
    right = xmax - x
    top = y - ymin
    bottom = ymax - y

    left = max(left, 0.0)
    right = max(right, 0.0)
    top = max(top, 0.0)
    bottom = max(bottom, 0.0)

    lr_min = min(left, right)
    lr_max = max(left, right)
    tb_min = min(top, bottom)
    tb_max = max(top, bottom)

    lr_ratio = lr_min / lr_max if lr_max != 0 else 1.0
    tb_ratio = tb_min / tb_max if tb_max != 0 else 1.0

    return math.sqrt(lr_ratio * tb_ratio)


def check_in(gt, answer):
    gt = gt.strip().lower()
    answer = answer.strip().lower()
    pattern = r'\b{}\b'.format(re.escape(gt))
    if re.search(pattern, answer):
        return True
    return False



_ARTICLE_RE = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)
def normalize(text: str) -> str:
    """
    Normalize the text by removing punctuation, articles, and extra spaces, and converting to lowercase.
    """
    text = text.strip().lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLE_RE.sub('', text)    
    text = ' '.join(text.split())          
    return text
def check_correct(answer, gt):
    return normalize(answer) == normalize(gt)


def main(args):

    eval_root_dir = Path(args.eval_root_dir)
    eval_model_path = args.eval_model_path
    vqas_dir = Path(args.vqas_dir)
    save_dir = Path(args.save_dir)

    # Get model paths
    model_paths = [eval_model_path] if eval_model_path.lower() != 'all' else [
        os.path.join(f, s) for f in os.listdir(eval_root_dir) for s in os.listdir(os.path.join(eval_root_dir, f))
    ]

    logger.info(f"Evaluating models: {model_paths}.")
    eval_results = []
    for model_path in model_paths:
        eval_result = []

        eval_dir = eval_root_dir / model_path
        logger.info(f"Processing model: {model_path}.")
        mllm_output_files = sorted(eval_dir.glob("*.json"))

        for mllm_output_file in mllm_output_files:
            with open(mllm_output_file, 'r') as file:
                mllm_output_data = json.load(file)
            logger.info(f"Load {len(mllm_output_data)} mllm output from {mllm_output_file}.")

            # Extract <answer> content
            match_mllm_output_list = []
            for data in mllm_output_data:
                try:
                    pattern = r'<answer>(.*?)</answer>'
                    match = re.search(pattern, data['output'], re.DOTALL)
                    if not match:
                        raise ValueError(f"No match found for pattern: {pattern}")
                    content = match.group(1).strip()
                    match_mllm_output_list.append(dict(vqa_idx=data['vqa_idx'], output=content, prompt=data['prompt']))
                except (TypeError, ValueError):
                    continue
            logger.info(f"Match {len(match_mllm_output_list)} mllm output.")

            mllm_output_file_name = mllm_output_file.name
            yaw_mark_in_name = 'yaw_vqas'
            xy2d_mark_in_name = 'xy2d_vqas'
            depth_mark_in_name = 'depth_vqas'
            dis_mark_in_name = 'dis_vqas'
            lr_mark_in_name = 'lr_vqas'
            fb_mark_in_name = 'fb_vqas'

            if yaw_mark_in_name in mllm_output_file_name:
                # get valid output
                valid_mllm_output_list = []
                for data in match_mllm_output_list:
                    try:    
                        yaw_output = data['output'].strip()
 
                        yaw_options_match = re.search(r'Options:.*?\n(.*?)(?=\n\n|\Z)', data['prompt'], re.DOTALL).group(1)
                        yaw_options = re.findall(r'-\s*(.*)', yaw_options_match)
                        in_option = False
                        for option in yaw_options:
                            if check_in(option, yaw_output):
                                in_option = True
                                break
                        assert in_option == True

                        valid_mllm_output_list.append(dict(vqa_idx=data['vqa_idx'], output=yaw_output))
                    except AssertionError:
                        continue
                logger.info(f"Get {len(valid_mllm_output_list)} valid vlm output.")

                # get GT
                yaw_answer_file = next(file for file in vqas_dir.glob("*.json") if yaw_mark_in_name in file.name)
                with open(yaw_answer_file, 'r') as file:
                    yaw_answer_data = json.load(file)
                logger.info(f"Load {len(yaw_answer_data)} yaw answer output from {yaw_answer_file}.")

                # handle output
                tmp_idx = 0
                correct_cnt = 0
                yaw_valid_num = 0
                yaw_qa_sum = len(yaw_answer_data) // 2
                for answer_idx in range(yaw_qa_sum):
                    for output_idx in range(tmp_idx, len(valid_mllm_output_list) - 1):
                        idx1 = 2*answer_idx
                        idx2 = 2*answer_idx+1
                        # Make sure both answers are valid
                        valid_output_idx1 = valid_mllm_output_list[output_idx]['vqa_idx']
                        valid_output_idx2 = valid_mllm_output_list[output_idx + 1]['vqa_idx']
                        if valid_output_idx1 == idx1 and valid_output_idx2 == idx2:
                            yaw_valid_num += 1

                            answer1 = yaw_answer_data[idx1]['answer']
                            answer2 = yaw_answer_data[idx2]['answer']
                            output1 = valid_mllm_output_list[output_idx]['output']
                            output2 = valid_mllm_output_list[output_idx + 1]['output']
                            if check_correct(answer1, output1) and check_correct(answer2, output2):
                                correct_cnt +=1

                            tmp_idx = output_idx # Save 10x + the time
                            break
                        
                        # Because it is orderly, it is not need to look for it later when it is greater than it is more
                        if valid_output_idx1 > idx1:
                            break

                # get score
                yaw_correct_rate = 0.0 if yaw_valid_num==0 else correct_cnt / yaw_valid_num
                yaw_score = correct_cnt / yaw_qa_sum
                yaw_valid_rate = yaw_valid_num / yaw_qa_sum
                eval_result.append(dict(
                    yaw_valid_response_accuracy=yaw_correct_rate, 
                    yaw_score=yaw_score, 
                    yaw_valid_response_rate=yaw_valid_rate, 
                    ))

            elif xy2d_mark_in_name in mllm_output_file_name:
                # get GT
                xy2d_answer_file = next(file for file in vqas_dir.glob("*.json") if xy2d_mark_in_name in file.name)
                with open(xy2d_answer_file, 'r') as file:
                    xy2d_answer_data = json.load(file)
                logger.info(f"Load {len(xy2d_answer_data)} yaw answer output from {xy2d_answer_file}.")
                image_width, image_height = map(int, xy2d_answer_data[0]['image_pixel'].split('x'))

                # get valid output
                valid_mllm_output_list = []
                for data in match_mllm_output_list:
                    try:    
                        xy2d_output = data['output'].strip()
                        
                        pattern = r'\[\s*([-\d.,\s]+)\s*\]'
                        match = re.search(pattern, xy2d_output)
                        if not match:
                            raise ValueError(f"No match found for pattern: {pattern}")
                        xy2d_output_list = ast.literal_eval(match.group(0))
                        
                        assert len(xy2d_output_list)==2 or len(xy2d_output_list)==4
                        if len(xy2d_output_list)==2:
                            x, y  = xy2d_output_list
                        else:
                            x1, y1, x2, y2 = xy2d_output_list
                            x = (x1+x2)//2
                            y = (y1+y2)//2
                        x = x * image_width if x < 1 else x
                        y = y * image_height if y <1 else y

                        assert image_width > x >= 0 and image_height > y >= 0
                        valid_mllm_output_list.append(dict(vqa_idx=data['vqa_idx'], output=[x, y]))
                    except (ValueError, AssertionError, SyntaxError) as e:
                        continue
                logger.info(f"Get {len(valid_mllm_output_list)} valid mllm output.")

                # handle output
                tmp_idx = 0
                score_cnt = 0
                xy2d_valid_num = 0
                xy2d_qa_sum = len(xy2d_answer_data) 
                for answer_idx in range(xy2d_qa_sum):
                    for output_idx in range(tmp_idx, len(valid_mllm_output_list)):
                        idx1 = answer_idx
                        valid_output_idx1 = valid_mllm_output_list[output_idx]['vqa_idx']

                        if valid_output_idx1 == idx1:
                            xy2d_valid_num += 1

                            obj_box = xy2d_answer_data[idx1]['obj_bbox']  
                            output1 = valid_mllm_output_list[output_idx]['output']

                            centerness_score = compute_centerness(output1, obj_box)
                            score_cnt += centerness_score

                            tmp_idx = output_idx 
                            break
                        
                        # Because it is orderly, it is not need to look for it later when it is greater than it is more
                        if valid_output_idx1 > idx1:
                            break
                # get score
                xy2d_correct_rate = 0.0 if xy2d_valid_num ==0 else score_cnt / xy2d_valid_num
                xy2d_score = score_cnt / xy2d_qa_sum
                xy2d_valid_rate = xy2d_valid_num / xy2d_qa_sum
                eval_result.append(dict(
                    xy2d_valid_response_accuracy=xy2d_correct_rate,
                    xy2d_score=xy2d_score,
                    xy2d_valid_response_rate=xy2d_valid_rate,
                    ))
            
            elif depth_mark_in_name in mllm_output_file_name:
                # get valid output
                valid_mllm_output_list = []
                for data in match_mllm_output_list:
                    try:    
                        depth_output = data['output'].strip()
                        depth_options_match = re.search(r'Options:.*?\n(.*?)(?=\n\n|\Z)', data['prompt'], re.DOTALL).group(1)
                        depth_options = re.findall(r'-\s*(.*)', depth_options_match)
                        in_option = False
                        for option in depth_options:
                            if check_in(option, depth_output):
                                in_option = True
                                break
                        assert in_option == True

                        valid_mllm_output_list.append(dict(vqa_idx=data['vqa_idx'], output=depth_output))
                    except (ValueError, AssertionError):
                        continue
                logger.info(f"Get {len(valid_mllm_output_list)} valid mllm output.")

                # get GT
                depth_answer_file = next(file for file in vqas_dir.glob("*.json") if depth_mark_in_name in file.name)
                with open(depth_answer_file, 'r') as file:
                    depth_answer_data = json.load(file)
                logger.info(f"Load {len(depth_answer_data)} depth answer output from {depth_answer_file}.")

                # handle output
                tmp_idx = 0
                correct_cnt = 0
                depth_valid_num = 0
                depth_qa_sum = len(depth_answer_data) 
                for answer_idx in range(depth_qa_sum):
                    for output_idx in range(tmp_idx, len(valid_mllm_output_list)):
                        idx1 = answer_idx
                        valid_output_idx1 = valid_mllm_output_list[output_idx]['vqa_idx']

                        if valid_output_idx1 == idx1:
                            depth_valid_num += 1
                            answer1 = depth_answer_data[idx1]['answer']   
                            output1 = valid_mllm_output_list[output_idx]['output']

                            if check_correct(answer1,output1):
                                correct_cnt +=1

                            tmp_idx = output_idx 
                            break
                        
                        # Because it is orderly, it is not need to look for it later when it is greater than it is more
                        if valid_output_idx1 > idx1:
                            break

                # get score
                depth_correct_rate = 0.0 if depth_valid_num==0 else correct_cnt / depth_valid_num
                depth_score = correct_cnt / depth_qa_sum
                depth_valid_rate = depth_valid_num / depth_qa_sum
                eval_result.append(dict(
                    depth_valid_response_accuracy=depth_correct_rate, 
                    depth_score=depth_score, 
                    depth_valid_response_rate=depth_valid_rate,
                    ))


            elif dis_mark_in_name in mllm_output_file_name:
                # get valid output
                valid_mllm_output_list = []
                for data in match_mllm_output_list:
                    try:    
                        dis_output = data['output'].strip()
                        dis_options_match = re.search(r'Options:.*?\n(.*?)(?=\n\n|\Z)', data['prompt'], re.DOTALL).group(1)
                        dis_options = re.findall(r'-\s*(.*)', dis_options_match)
                        in_option = False
                        for option in dis_options:
                            if check_in(option,dis_output):
                                in_option = True
                                break
                        assert in_option == True

                        valid_mllm_output_list.append(dict(vqa_idx=data['vqa_idx'], output=dis_output))
                    except (ValueError,AssertionError):
                        continue
                logger.info(f"Get {len(valid_mllm_output_list)} valid mllm output.")

                # get GT
                dis_answer_file = next(file for file in vqas_dir.glob("*.json") if dis_mark_in_name in file.name)
                with open(dis_answer_file, 'r') as file:
                    dis_answer_data = json.load(file)
                logger.info(f"Load {len(dis_answer_data)} dis answer output from {dis_answer_file}.")

                # handle output
                tmp_idx = 0
                correct_cnt = 0
                dis_valid_num = 0
                dis_qa_sum = len(dis_answer_data) // 2
                for answer_idx in range(dis_qa_sum):
                    for output_idx in range(tmp_idx, len(valid_mllm_output_list) - 1):

                        idx1 = 2*answer_idx
                        idx2 = 2*answer_idx+1
                        # Make sure both answers are valid
                        valid_output_idx1 = valid_mllm_output_list[output_idx]['vqa_idx']

                        valid_output_idx2 = valid_mllm_output_list[output_idx + 1]['vqa_idx']
                        if valid_output_idx1 == idx1 and valid_output_idx2 == idx2:
                            dis_valid_num += 1

                            answer1 = dis_answer_data[idx1]['answer']
                            answer2 = dis_answer_data[idx2]['answer']
                            output1 = valid_mllm_output_list[output_idx]['output']
                            output2 = valid_mllm_output_list[output_idx + 1]['output']
                            if check_correct(answer1, output1) and check_correct(answer2, output2):
                                correct_cnt +=1


                            tmp_idx = output_idx
                            break
                        
                        # Because it is orderly, it is not need to look for it later when it is greater than it is more
                        if valid_output_idx1 > idx1:
                            break

                # get score
                dis_correct_rate = 0.0 if dis_valid_num ==0 else correct_cnt / dis_valid_num
                dis_score = correct_cnt / dis_qa_sum
                dis_valid_rate = dis_valid_num / dis_qa_sum
                eval_result.append(dict(
                    dis_valid_response_accuracy=dis_correct_rate, 
                    dis_score=dis_score, 
                    dis_valid_response_rate=dis_valid_rate, 
                    ))

            elif "lr_vqas" in mllm_output_file_name:
                # get valid output
                valid_mllm_output_list = []
                for data in match_mllm_output_list:
                    try:    
                        lr_output = data['output'].strip()

                        lr_options_match = re.search(r'Options:.*?\n(.*?)(?=\n\n|\Z)', data['prompt'], re.DOTALL).group(1)
                        lr_options = re.findall(r'-\s*(.*)', lr_options_match)
                        in_option = False
                        for option in lr_options:
                            if check_in(option,lr_output):
                                in_option = True
                                break
                        assert in_option == True

                        valid_mllm_output_list.append(dict(vqa_idx=data['vqa_idx'], output=lr_output))
                    except (ValueError,AssertionError):
                        continue
                logger.info(f"Get {len(valid_mllm_output_list)} valid mllm output.")

                # get GT
                lr_answer_file = next(file for file in vqas_dir.glob("*.json") if lr_mark_in_name in file.name)
                with open(lr_answer_file, 'r') as file:
                    lr_answer_data = json.load(file)
                logger.info(f"Load {len(lr_answer_data)} lr answer output from {lr_answer_file}.")

                # handle output
                tmp_idx = 0
                correct_cnt = 0
                lr_valid_num = 0
                lr_qa_sum = len(lr_answer_data) // 2
                for answer_idx in range(lr_qa_sum):
                    for output_idx in range(tmp_idx, len(valid_mllm_output_list) - 1):
                        idx1 = 2*answer_idx
                        idx2 = 2*answer_idx+1
                        # Make sure both answers are valid
                        valid_output_idx1 = valid_mllm_output_list[output_idx]['vqa_idx']
                        valid_output_idx2 = valid_mllm_output_list[output_idx + 1]['vqa_idx']
                        if valid_output_idx1 == idx1 and valid_output_idx2 == idx2:
                            lr_valid_num += 1

                            answer1 = lr_answer_data[idx1]['answer']
                            answer2 = lr_answer_data[idx2]['answer']
                            output1 = valid_mllm_output_list[output_idx]['output']
                            output2 = valid_mllm_output_list[output_idx + 1]['output']
                            if check_correct(answer1,output1) and check_correct(answer2,output2):
                                correct_cnt +=1

                            tmp_idx = output_idx
                            break
                        
                        # Because it is orderly, it is not need to look for it later when it is greater than it is more
                        if valid_output_idx1 > idx1:
                            break
                
                # get score
                lr_correct_rate = 0.0 if lr_valid_num==0 else correct_cnt / lr_valid_num
                lr_score = correct_cnt / lr_qa_sum
                lr_valid_rate = lr_valid_num / lr_qa_sum
                eval_result.append(dict(
                    lr_valid_response_accuracy=lr_correct_rate, 
                    lr_score=lr_score, 
                    lr_valid_response_rate=lr_valid_rate, 
                    ))
            
            elif fb_mark_in_name in mllm_output_file_name:
                # get valid output
                valid_mllm_output_list = []
                for data in match_mllm_output_list:
                    try:    
                        fb_output = data['output'].strip()
                        fb_options_match = re.search(r'Options:.*?\n(.*?)(?=\n\n|\Z)', data['prompt'], re.DOTALL).group(1)
                        fb_options = re.findall(r'-\s*(.*)', fb_options_match)
                        in_option = False
                        for option in fb_options:
                            if check_in(option, fb_output):
                                in_option = True
                                break
                        assert in_option == True
                    
                        valid_mllm_output_list.append(dict(vqa_idx=data['vqa_idx'], output=fb_output))
                    except (ValueError,AssertionError):
                        continue
                logger.info(f"Get {len(valid_mllm_output_list)} valid mllm output.")

                # get GT
                fb_answer_file = next(file for file in vqas_dir.glob("*.json") if fb_mark_in_name in file.name)
                with open(fb_answer_file, 'r') as file:
                    fb_answer_data = json.load(file)
                logger.info(f"Load {len(fb_answer_data)} fb answer output from {fb_answer_file}.")

                # handle output
                tmp_idx = 0
                correct_cnt = 0
                fb_valid_num = 0
                fb_qa_sum = len(fb_answer_data) // 2
                for answer_idx in range(fb_qa_sum):
                    for output_idx in range(tmp_idx, len(valid_mllm_output_list) - 1):
                        idx1 = 2*answer_idx
                        idx2 = 2*answer_idx+1
                        # Make sure both answers are valid
                        valid_output_idx1 = valid_mllm_output_list[output_idx]['vqa_idx']
                        valid_output_idx2 = valid_mllm_output_list[output_idx + 1]['vqa_idx']
                        if valid_output_idx1 == idx1 and valid_output_idx2 == idx2:
                            fb_valid_num += 1
                            answer1 = fb_answer_data[idx1]['answer']
                            answer2 = fb_answer_data[idx2]['answer']
                            output1 = valid_mllm_output_list[output_idx]['output']
                            output2 = valid_mllm_output_list[output_idx + 1]['output']
                            if check_correct(answer1,output1) and check_correct(answer2,output2):
                                correct_cnt +=1

                            tmp_idx = output_idx
                            break
                        
                        # Because it is orderly, it is not need to look for it later when it is greater than it is more
                        if valid_output_idx1 > idx1:
                            break
                # get score
                fb_correct_rate = 0.0 if fb_valid_num ==0 else correct_cnt / fb_valid_num
                fb_score = correct_cnt / fb_qa_sum
                fb_valid_rate = fb_valid_num / fb_qa_sum
                eval_result.append(dict(
                    fb_valid_response_accuracy=fb_correct_rate, 
                    fb_score=fb_score, 
                    fb_valid_response_rate=fb_valid_rate, 
                    ))

        # Save evaluation results to CSV
        save_path  = save_dir / model_path
        save_path.mkdir(exist_ok=True, parents =True)
        df = pd.DataFrame(eval_result)
        df.to_csv(save_path / f'eval_result.csv', index=False)

        eval_results.append(eval_result)

    # Aggregate final metrics for all models
    for idx, results in enumerate(eval_results):
        if len(results) ==0:
            continue
        
        print(model_paths[idx])
        valid_response_accuracy = 0
        score= 0
        valid_response_rate = 0
        for result in results:
            for k in result.keys():
                if "valid_response_accuracy" in k:
                    valid_response_accuracy += result[k]
                if "score" in k:
                    score += result[k]
                if "valid_response_rate" in k:
                    valid_response_rate += result[k]
            print(result)
        # assert len(results)==6
        valid_response_accuracy /= len(results)
        score /= len(results)
        valid_response_rate /= len(results)
        print(dict(valid_response_accuracy=valid_response_accuracy,score=score,valid_response_rate=valid_response_rate))
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM output.")
    parser.add_argument('--vqas_dir', type=str, default='',
                        help='Specify the folder for the VQA to use for evaluation.')
    parser.add_argument('--eval_root_dir', type=str, default='',
                        help='Specify the root directory for VLM output files.')
    parser.add_argument('--eval_model_path', type=str, default='',
                        help='Specify the path of the model to evaluate.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='Specify the directory where evaluation results will be saved.')
    args = parser.parse_args()
    main(args)
