import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import json
import math
from segment_anything.datasets.utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON file: {file_path}")

def generate_image_dict(omni3d_json):
    """生成image字典和id到路径的映射"""
    image_dict = {}
    id2path = {}
    for image in omni3d_json['images']:
        file_path = image['file_path'].replace('//', '/')
        image_dict[file_path] = {
            'obj_list': [],
            'pred_obj_list': [],
            'K': image['K']
        }
        id2path[image['id']] = file_path
    return image_dict, id2path

def process_annotations(annotations, id2path, image_dict):
    """处理标注数据"""
    for anno in annotations:
        path = id2path[anno['image_id']]
        image_dict[path]['obj_list'].append(anno)

def process_predictions(pred_json, id2path, image_dict):
    """处理预测数据"""
    for pred_instance in pred_json:
        path = id2path[pred_instance['image_id']]
        image_dict[path]['pred_obj_list'].append(pred_instance)

def generate_sample(instance, path, img_path, depth_path, path2id, pred_mode):
    """生成单个样本的字典"""
    obj_list = []
    todo_obj_list = instance['pred_obj_list'] if pred_mode else instance['obj_list']
    if not todo_obj_list:
        return None

    for obj in todo_obj_list:
        x, y, z = obj['center_cam']
        w, h, l = obj['dimensions']
        # print(obj)
        if 'R_cam' in obj:
            pose = np.array(obj['R_cam'])
        elif 'pose' in obj:
            pose = np.array(obj['pose'])
        else:
            print(obj)
            print(f"Warning: Neither 'R_cam' nor 'pose' found for object in {path}")
            continue
        # pose = np.array(obj.get('R_cam', obj['pose']))
        yaw = math.atan2(pose[0, 0], pose[2, 0])

        # 旋转矩阵调整
        R_90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        pose = np.dot(pose, R_90)

        # 2D边界框和其他信息
        if pred_mode:
            bbox_2d_proj = obj['bbox']
            bbox_2d_tight = bbox_2d_trunc = [-1, -1, -1, -1]
            score = obj['score']
            visibility = truncation = -1
        else:
            bbox_2d_proj = obj['bbox2D_proj']
            bbox_2d_tight = obj['bbox2D_tight']
            bbox_2d_trunc = obj['bbox2D_trunc']
            score = 1
            visibility = obj['visibility']
            truncation = obj['truncation']

        label = obj['category_id']
        image_id = path2id[path]
        instance_id = generate_instance_id(obj, img_path)

        obj_list.append({
            "3d_bbox": [x, y, z, w, h, l, yaw],
            "2d_bbox_proj": bbox_2d_proj,
            "2d_bbox_tight": bbox_2d_tight,
            "2d_bbox_trunc": bbox_2d_trunc,
            "label": label,
            "rotation_pose": pose,
            "instance_id": instance_id,
            "score": score,
            "image_id": image_id,
            "visibility": visibility,
            "truncation": truncation,
        })

    if not obj_list:
        return None

    return {
        "img_path": img_path,
        "depth_path": depth_path,
        "K": np.array(instance['K']).reshape(1, 3, 3),
        "obj_list": obj_list,
    }

def process_image(path, instance, path2id, pred_mode, dataset_name='kitti'):
    """
    Processes image and depth paths based on dataset name.
    Supports: kitti, nuscenes, objectron, arkitscenes, sunrgbd, hypersim, waymo, cityscapes3d, 3rscan
    """
    import os

    dataset_path_map = {
        'kitti': lambda p: './data/kitti' + p.split('KITTI_object')[-1],
        'nuscenes': lambda p: './data/nuscenes' + p.split('nuScenes')[-1],
        'objectron': lambda p: './data/objectron' + p.split('objectron')[-1],
        'arkitscenes': lambda p: './data/ARKitScenes' + p.split('ARKitScenes')[-1],
        'sunrgbd': lambda p: './data/SUNRGBD' + p.split('SUNRGBD')[-1],
        'hypersim': lambda p: './data/hypersim' + p.split('hypersim')[-1],
        'waymo': lambda p: './data/waymo' + p.split('waymo')[-1],
        'cityscapes3d': lambda p: './data/cityscapes3d' + p.split('cityscapes3d')[-1],
        '3rscan': lambda p: './data/3RScan' + p.split('3RScan')[-1],
    }

    if dataset_name not in dataset_path_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    img_path = dataset_path_map[dataset_name](path)

    if not os.path.exists(img_path):
        print(f"Image path does not exist: {img_path}")
        return None

    # Depth path logic
    if dataset_name == 'kitti':
        tmp_path = img_path.replace('.png', '_depth.npy').split('/')
        depth_path = './data/kitti/test_depth_front/' + '/'.join(tmp_path[-2:])
    elif dataset_name == 'nuscenes':
        depth_path = './data/nuscenes/nuscenes_depth/' + img_path.split('samples')[-1].replace('.jpg', '.png')
    elif dataset_name == 'hypersim':
        parts = path.split('/')
        depth_path = f'./data/hypersim/depth_in_meter/{parts[1]}/images/{parts[3].replace("final_preview", "geometry_hdf5")}/{parts[-1].replace("tonemap.jpg", "depth_meters.hdf5")}'
    elif dataset_name == 'sunrgbd':
        tmp_path = path.split('SUNRGBD')[-1]
        data_name = tmp_path.split('/image/')[0]
        base_path = f'./data/SUNRGBD{data_name}/depth_bfx/'
        tail_name = os.listdir(base_path)[0]
        depth_path = base_path + tail_name
    else:
        depth_path = None  # For objectron, arkitscenes, etc.

    if depth_path and not os.path.exists(depth_path):
        print(f"Depth path does not exist: {depth_path}")
        return None

    return generate_sample(instance, path, img_path, depth_path, path2id, pred_mode)

def process_data(output_pkl, dataset_name, omni3d_json_path=None, pred_omni3d_json_path=None, max_workers=8):
    """主函数：处理数据并保存"""
    omni3d_json = load_json(omni3d_json_path)
    image_dict, id2path = generate_image_dict(omni3d_json)
    process_annotations(omni3d_json['annotations'], id2path, image_dict)

    pred_mode = pred_omni3d_json_path is not None
    if pred_mode:
        print("Processing Cube R-CNN output as 3DAW input.")
        pred_omni3d_json = load_json(pred_omni3d_json_path)
        process_predictions(pred_omni3d_json, id2path, image_dict)

    path2id = {v: k for k, v in id2path.items()}
    samples = []

    # 使用多线程处理每张图片
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for path, instance in image_dict.items():
            futures.append(executor.submit(process_image, path, instance, path2id, pred_mode, dataset_name))

        # 处理返回结果
        for future in tqdm(as_completed(futures), desc="Processing images"):
            sample = future.result()
            if sample:
                samples.append(sample)

    print(f"Processed {len(samples)} samples.")
    with open(output_pkl, 'wb') as f:
        pickle.dump(samples, f)

# 配置路径
index = 1
mode = 'test'

dataset_name_test = ['ARKitScenes_test_novel', 'KITTI_test_novel', 'SUNRGBD_test_novel', '3RScan_test_novel', 'Waymo_test']
dataset_keys = [ 'arkitscenes', 'kitti', 'sunrgbd', '3rscan', 'waymo']

# 根据mode选择数据集
if mode == 'test':
    dataset_list = dataset_name_test
else:
    raise ValueError(f"Unknown mode: {mode}")

# 选择数据集名称
selected_dataset_name = dataset_list[index]
selected_dataset_key = dataset_keys[index]


# pred_omni3d_json_path = f'/cpfs01/user/zhanghanxue/segment-anything/data/ovmono3d_pred/{selected_dataset_name}/omni_instances_results.json'
pred_omni3d_json_path = None


output_pkl = f'./data/pkls/ovmono3d_novel_gt/{selected_dataset_name}.pkl'
omni3d_json_path = f'/cpfs01/user/zhanghanxue/omni3d/datasets/Omni3D/{selected_dataset_name}.json'



process_data(output_pkl, selected_dataset_key, omni3d_json_path, pred_omni3d_json_path)