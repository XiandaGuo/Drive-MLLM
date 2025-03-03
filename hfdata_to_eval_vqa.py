import logging
from datasets import load_dataset
import random
from prompt.prompt_manager import PromptManager
from pathlib import Path
import json
from tqdm import tqdm
import argparse

def format_range(input_list):
    if len(input_list) == 2:
        return f"Between {input_list[0]} {'meter' if input_list[0] <= 1 else 'meters'} and {input_list[1]} {'meter' if input_list[1] <= 1 else 'meters'}"
    else:
        raise ValueError("Range fotmat error")

def generate_depth_range(depth):
    answer = depth
    if answer < 7:
        # 6.9 -> (3, 10) (12, 16) (18, 22)
        answer_len = random.uniform(6, 7)
        answer_range = [max(1, round(answer - answer_len / 2)), round(answer + answer_len / 2)]

        range2_start = answer_range[1] + random.randint(1, 2)
        range2_end = range2_start + random.randint(3, 4)
        range3_start = range2_end + random.randint(1, 2)
        range3_end = range3_start + random.randint(3, 4)
        range2 = [range2_start, range2_end]
        range3 = [range3_start, range3_end]

    elif answer > 15:
        # 15.1 -> (12, 19) (6, 10) (1, 4)
        answer_len = random.uniform(6, 7)
        answer_range = [round(answer - answer_len / 2), round(answer + answer_len / 2)]

        range2_end = answer_range[0] - random.randint(1, 2)
        range2_start = range2_end - random.randint(3, 4)
        range3_end = range2_start - random.randint(1, 2)
        range3_start = max(1, range3_end - random.randint(3, 4))
        range2 = [range2_start, range2_end]
        range3 = [range3_start, range3_end]

    else:
        # 7 - 15
        # 15.1 -> (12, 19) (6, 10) (1, 4)
        answer_len = random.uniform(6, 7)
        answer_range = [round(answer - answer_len / 2), round(answer + answer_len / 2)]

        range2_end = answer_range[0] - random.randint(1, 2)
        range2_start = max(1, range2_end - random.randint(3, 4))

        range3_start = answer_range[1] + random.randint(1, 2)
        range3_end = range3_start + random.randint(3, 4)

        range2 = [range2_start, range2_end]
        range3 = [range3_start, range3_end]

    return answer_range, range2, range3


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):

    # Set a random seed for reproducibility
    random.seed(42)
    
    # Load dataset from Hugging Face
    hf_dataset = args.hf_dataset
    logger.info(f"Loading Dataset...")
    dataset = load_dataset(hf_dataset, split='validation')
    dataset_num = len(dataset) # len(dataset)
    logger.info(f"Dataset loaded successfully with {dataset_num} data.")

    # Get image_name from dataset
    image_names = [Path(k).name for k in dataset.info.download_checksums.keys() if Path(k).suffix == '.webp']
    assert len(image_names) == dataset_num
    
    # prompt manager
    prm_mgr = PromptManager()

    # create dir
    vqas_save_dir = Path(config.vqas_save_dir)
    vqas_save_dir.mkdir(exist_ok=True)
    image_save_dir = vqas_save_dir / "image"
    image_save_dir.mkdir(exist_ok=True)

    # Handle multi object sample
    dis_vqa, lr_vqa, fb_vqa = [], [], []
    multi_obj_sample_idx = []
    for data_index in tqdm(range(dataset_num), desc="Handle multi object data:"):

        # attributes
        data = dataset[data_index]
        image = data['image']
        obj_descs = data['descs']
        obj_dists = data['distances']
        obj_xy2Ds = data['xy2Ds']
        obj_depths = data['depths']

        # image save
        image_save_path = str(image_save_dir / image_names[data_index])
        image.save(image_save_path)
        width, height = image.size

        # iter multi object
        obj_num = len(obj_descs)
        if obj_num > 1:
            multi_obj_sample_idx.append(data_index)
            for idx1 in range(obj_num - 1):
                for idx2 in range(idx1 + 1 ,obj_num):
                    # object
                    obj1 = 'the ' + obj_descs[idx1]
                    obj2 = 'the ' + obj_descs[idx2]

                    # dis vqa
                    dis_c_vqa_prompt = prm_mgr.dis_vqa_template.format(obj1, obj2, "closer", obj1.capitalize(), obj2.capitalize(), *prm_mgr.markers)
                    dis_f_vqa_prompt = prm_mgr.dis_vqa_template.format(obj1, obj2, "farther", obj1.capitalize(), obj2.capitalize(), *prm_mgr.markers)
                    obj1_dis = obj_dists[idx1]
                    obj2_dis = obj_dists[idx2]
                    dis_threshold = 1
                    if abs(obj1_dis - obj2_dis) <= dis_threshold:
                        dis_c_answer = "Almost the same"
                        dis_f_answer = "Almost the same"
                    elif obj1_dis < obj2_dis:
                        dis_c_answer = obj1.capitalize()
                        dis_f_answer = obj2.capitalize()
                    else:
                        dis_c_answer = obj2.capitalize()
                        dis_f_answer = obj1.capitalize()
                    dis_c_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", prompt=dis_c_vqa_prompt, answer=dis_c_answer)
                    dis_f_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", prompt=dis_f_vqa_prompt, answer=dis_f_answer)
                    dis_vqa.append(dis_c_save_dict)
                    dis_vqa.append(dis_f_save_dict)
                    
                    # lr vqa
                    l_vqa_prompt = prm_mgr.lr_vqa_template.format('left', obj1, obj2, obj1.capitalize(), obj2.capitalize(), *prm_mgr.markers)
                    r_vqa_prompt = prm_mgr.lr_vqa_template.format('right', obj1, obj2, obj1.capitalize(), obj2.capitalize(), *prm_mgr.markers)
                    obj1_x = obj_xy2Ds[idx1][0]
                    obj2_x = obj_xy2Ds[idx2][0]
                    x_threshold = 100
                    if abs(obj1_x - obj2_x) < x_threshold:
                        l_answer = "Almost the same"
                        r_answer = "Almost the same"
                    elif obj1_x < obj2_x:
                        l_answer = obj1.capitalize()
                        r_answer = obj2.capitalize()
                    else:
                        l_answer = obj2.capitalize()
                        r_answer = obj1.capitalize()
                    l_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", prompt=l_vqa_prompt, answer=l_answer)
                    r_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", prompt=r_vqa_prompt, answer=r_answer)
                    lr_vqa.append(l_save_dict)
                    lr_vqa.append(r_save_dict)
                    
                    # fb vqa
                    f_vqa_prompt = prm_mgr.fb_vqa_template.format(obj1, "in front of", obj2, *prm_mgr.markers)
                    b_vqa_prompt = prm_mgr.fb_vqa_template.format(obj1, "behind", obj2, *prm_mgr.markers)
                    obj1_d = obj_depths[idx1]
                    obj2_d = obj_depths[idx2]
                    d_threshold = 0.5
                    if abs(obj1_d - obj2_d) < d_threshold:
                        f_answer = "Almost the same in terms of front-back position"
                        b_answer = "Almost the same in terms of front-back position"
                    elif obj1_d > obj2_d:
                        f_answer = 'Yes'
                        b_answer = 'No'
                    else:
                        f_answer = 'No'
                        b_answer = 'Yes'
                    f_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", prompt=f_vqa_prompt, answer=f_answer)
                    b_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", prompt=b_vqa_prompt, answer=b_answer)
                    fb_vqa.append(f_save_dict)
                    fb_vqa.append(b_save_dict)

    # save to json
    dis_vqas_save_json = vqas_save_dir / "03_dis_vqas.json"
    with open(str(dis_vqas_save_json), 'w') as file:
        json.dump(dis_vqa, file, indent=4)
    lr_vqas_save_json = vqas_save_dir / "04_lr_vqas.json"
    with open(str(lr_vqas_save_json), 'w') as file:
        json.dump(lr_vqa, file, indent=4)
    fb_vqas_save_json = vqas_save_dir / "05_fb_vqas.json"
    with open(str(fb_vqas_save_json), 'w') as file:
        json.dump(fb_vqa, file, indent=4)

    # get single object num
    assert len(dis_vqa) == len(lr_vqa) == len(fb_vqa)
    multi_obj_vqa_num = len(dis_vqa) // 2
    logger.info(f"Get {multi_obj_vqa_num} multi object vqas from {len(multi_obj_sample_idx)} samples.")
    single_obj_sample_idxs = [i for i in range(dataset_num) if i not in multi_obj_sample_idx]
    chosen_single_obj_sample_idxs = random.sample(single_obj_sample_idxs, multi_obj_vqa_num)
    chosen_single_obj_sample_idxs.sort()

    # Handle single object sample
    logger.info(f"Get same num single object vqas from {len(single_obj_sample_idxs)} data.")
    yaw_vqa, xy2d_vqa, depth_vqa = [], [], []
    for idx in tqdm(range(multi_obj_vqa_num), desc="Handle single object data:"):
        data_index = chosen_single_obj_sample_idxs[idx]
        
        # attributes
        data = dataset[data_index]
        image = data['image']
        bboxes2D = data['bboxes2D']
        bbox = bboxes2D[0]
        width, height = image.size
        obj_descs = data['descs']
        obj_yaw_descs = data['yaw_descs']
        obj_xy2ds = data['xy2Ds']
        obj_depths = data['depths']
        assert len(obj_descs) == 1

        # image save
        image_save_path = str(image_save_dir / image_names[data_index])
        if not Path(image_save_path).exists():
            image.save(image_save_path)

        
        # yaw vqa
        obj = 'the ' + obj_descs[0]
        obj_yaw_desc = obj_yaw_descs[0]
        yaw_option1s = random.sample(['Northeast', 'Southeast', 'Northwest', 'Southwest'], k=4)
        yaw_option2s = random.sample(['East', 'South', 'West', 'North'], k=4)
        if obj_yaw_desc in yaw_option1s:
            yaw_options = yaw_option1s
        elif obj_yaw_desc in yaw_option2s:
            yaw_options = yaw_option2s
        else:
            raise ValueError("Yaw option error")
        n_yaw_answer = obj_yaw_desc
        n_yaw_vqa_prompt = prm_mgr.yaw_vqa_template.format("North", obj , *yaw_options, *prm_mgr.markers)
        n_yaw_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", obj_bbox=bbox, prompt=n_yaw_vqa_prompt, answer=n_yaw_answer)
        opposite_map = {
            'North': 'South', 'South': 'North', 'East': 'West', 'West': 'East',
            'Northeast': 'Southwest', 'Southeast': 'Northwest', 'Southwest': 'Northeast', 'Northwest': 'Southeast'
        }
        s_yaw_answer = opposite_map[obj_yaw_desc]
        s_yaw_vqa_prompt = prm_mgr.yaw_vqa_template.format("South", obj , *yaw_options, *prm_mgr.markers)
        s_yaw_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", obj_bbox=bbox, prompt=s_yaw_vqa_prompt, answer=s_yaw_answer)
        yaw_vqa.append(n_yaw_save_dict)
        yaw_vqa.append(s_yaw_save_dict)

        # xy2d vqa
        xy2d_vqa_prompt = prm_mgr.xy2d_vqa_template.format(obj, *prm_mgr.markers)
        obj_xy2d = obj_xy2ds[0]
        xy2d_answer = str(obj_xy2d)
        xy2d_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", obj_bbox=bbox, prompt=xy2d_vqa_prompt, answer=xy2d_answer)
        xy2d_vqa.append(xy2d_save_dict)

        # depth vqa
        obj_depth = obj_depths[0]
        answer_option, depth_option1, depth_option2 = generate_depth_range(obj_depth)
        depth_options = random.sample([format_range(answer_option), format_range(depth_option1), format_range(depth_option2)], k=3)
        depth_vqa_prompt = prm_mgr.depth_vqa_template.format(obj, *depth_options, *prm_mgr.markers)
        depth_answer = format_range(answer_option)
        depth_save_dict = dict(image_path=image_save_path, image_pixel=f"{width}x{height}", obj_bbox=bbox, prompt=depth_vqa_prompt, answer=depth_answer)
        depth_vqa.append(depth_save_dict)

    # save
    yaw_vqas_save_json = vqas_save_dir / "00_yaw_vqas.json"
    with open(str(yaw_vqas_save_json), 'w') as file:
        json.dump(yaw_vqa, file, indent=4)
    xy2d_vqas_save_json = vqas_save_dir / "01_xy2d_vqas.json"
    with open(str(xy2d_vqas_save_json), 'w') as file:
        json.dump(xy2d_vqa, file, indent=4)
    depth_vqas_save_json = vqas_save_dir / "02_depth_vqas.json"
    with open(str(depth_vqas_save_json), 'w') as file:
        json.dump(depth_vqa, file, indent=4)

        
       

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate vqas from Hugging Face hub.")

    parser.add_argument('--hf_dataset', type=str, default='bonbon-rj/DriveMLLM',
                        help='Specify the path to the Hugging Face dataset.')
    parser.add_argument('--vqas_save_dir', type=str, default='eval_vqas',
                        help='Define the directory where the vqas files will be saved.')

    args = parser.parse_args()
    main(args)




