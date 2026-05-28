from torch.utils.data import Dataset
import torch

import torch.nn.functional as F
import cv2
import pickle
import numpy as np
from PIL import Image

from detect_anything.utils.transforms import ResizeLongestSide
from copy import deepcopy
from detect_anything.datasets.utils import *
from shapely.geometry import MultiPoint
from shapely.geometry import box

import matplotlib
import os
import json
import math
import random
import h5py

from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert


class DetAny3DDataset(Dataset):

    def __init__(self, 
                 cfg,
                 transform,
                 mode,
                 # for val different dataset respectively
                 dataset_name=None,
                 ):

        self.dataset_name_list = []
        self.pkl_path_list = []
        self.len_idx = []
        self.pkl_list = []

        self.sam_trans = ResizeLongestSide(cfg.model.pad)
        self.transform = transform

        if mode == 'train':
            self.dataset_dict = cfg.dataset.train
        elif mode == 'val':
            self.dataset_dict = cfg.dataset.val
        else:
            raise NotImplementedError('no test mode yet')
        
        if dataset_name is not None:
            # 仅加载指定数据集 (val)
            dataset_info = self.dataset_dict[dataset_name]
            self._load_single_dataset(dataset_name, dataset_info)
        else:
            # 加载所有数据集 (train)
            for dataset_name in self.dataset_dict.keys():
                dataset_info = self.dataset_dict[dataset_name]
                self._load_single_dataset(dataset_name, dataset_info)
        self.raw_info = [(dataset_name, num_samples) for dataset_name, num_samples in zip(self.dataset_name_list, self.len_idx)]

        self.idx_cum = np.cumsum(self.len_idx)
        self.pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
        self.cfg = cfg
        self.mode = mode

        if self.mode == 'val' and self.cfg.dataset.dino_as_input:
            self.dino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "GroundingDINO/weights/groundingdino_swint_ogc.pth")
            self.BOX_TRESHOLD = 0.001
            self.TEXT_TRESHOLD = 0.25
        with open('./data/category_meta.json', 'r') as f:
            self.category_id = json.load(f)

    def _load_single_dataset(self, dataset_name, dataset_info):
        """加载单个数据集的功能，供重复使用"""
        self.dataset_name_list.append(dataset_name)
        self.pkl_path_list.append(dataset_info.pkl_path)
        
        with open(dataset_info.pkl_path, 'rb') as f:
            tmp_pkl = pickle.load(f)[dataset_info.range.begin:dataset_info.range.end:dataset_info.range.interval]
            self.pkl_list.append(tmp_pkl)
        self.len_idx.append(len(tmp_pkl))
        print(f"Dataset: {dataset_name}, Number of samples: {len(tmp_pkl)}")
        # temp add coco here, todo: change to general
        if dataset_name == 'coco':
            from segment_anything.datasets.coco_utils import COCO
            self.annFile = '{}/annotations/instances_{}.json'.format(dataset_info.dataDir, dataset_info.dataType)
            self.coco = COCO(self.annFile)
            self.data_list = self.coco.getImgIds()
        if 'A2D2' in dataset_name:
            with open ('./data/A2D2/cams_lidars.json', 'r') as f:
                self.A2D2_config = json.load(f)

    def _get_relative_index(self, index):

        idx = 0
        for i, i_len in enumerate(self.idx_cum):
            if index >= i_len:
                idx = i + 1
        if idx > 0:
            true_index = index - self.idx_cum[idx - 1]
        else:
            true_index = index

        return idx, true_index
    
    def _load_depth(self, depth_path, dataset_name, img):
        # import ipdb;ipdb.set_trace()
        if depth_path is None:
            height, width = img.shape[:2]
            depth = np.zeros((height, width), dtype=np.float32)
        elif depth_path[-4:] == '.png':
            depth = np.array(Image.open(depth_path)).astype(np.float32)
            depth = depth / self.dataset_dict[dataset_name].metric_scale
        elif depth_path[-4:] == '.npy':
            depth = np.load(depth_path).astype(np.float32)
        elif depth_path[-4:] == 'hdf5':
            assert 'hypersim' in dataset_name, 'only hypersim support now'
            intWidth = 1024
            intHeight = 768
            fltFocal = 886.81

            hf = h5py.File(depth_path, 'r')
            n1 = hf.get('dataset')[:]
            npyDistance = np.array(n1)

            npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
            npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
            npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
            npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

            depth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
            depth = depth.astype(np.float32)
        else:
            raise NotImplementedError

        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        return depth

    def __getitem__(self, index):

        idx, true_index = self._get_relative_index(index)
        pkl_now = self.pkl_list[idx]
        dataset_name = self.dataset_name_list[idx]
        
        instance = pkl_now[true_index]
        K = instance['K'].astype(np.float32)
        K = torch.tensor(K)

        img_path = instance['img_path']

        if self.cfg.dataset.hack_img_path:
            img_path = self.cfg.dataset.hack_img_path
        if not os.path.exists(img_path):
            print(f"img_path {img_path} not exists")
            return self.__getitem__(random.randint(0, self.idx_cum[-1]-1)) 

        todo_img = cv2.imread(img_path)
        if 'A2D2' in dataset_name:
            todo_img = undistort_image(todo_img, 'front_center', self.A2D2_config)
        
        todo_img = cv2.cvtColor(todo_img, cv2.COLOR_BGR2RGB)
        original_size = tuple(todo_img.shape[:-1])

        depth_path = instance['depth_path']
        
        if self.cfg.dataset.hack_img_path:
            depth_path = None
        depth = self._load_depth(depth_path, dataset_name, todo_img)
        
        img, depth = self.transform(todo_img, depth)
        
        cropped_size = tuple(img.shape[1:3])
        cropped_blank_H = int((original_size[0] - cropped_size[0]) / 2)
        cropped_blank_W = int((original_size[1] - cropped_size[1]) / 2)
        
        # bx, by will change if cropped
        K[0, 0, 2] = K[0, 0, 2] - cropped_blank_W
        K[0, 1, 2] = K[0, 1, 2] - cropped_blank_H

        # resize the long edge to target size
        img = img.unsqueeze(0)
        img = self.sam_trans.apply_image_torch(img)
        
        before_pad_size = tuple(img.shape[-2:])
        resize_ratio = before_pad_size[1] / cropped_size[1]

        # fx, fy, bx, by will change if resized
        K[0, 0] = K[0, 0] * resize_ratio
        K[0, 1] = K[0, 1] * resize_ratio

        depth = self.sam_trans.apply_depth_torch(depth.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
       
        # insure the short edge is divisible by 112
        before_crop_size = tuple(img.shape[2:])
        img, depth, K = self.crop_hw(img, depth, K)
        before_pad_size = tuple(img.shape[2:])
        
        raw_image = img.clone().squeeze(0)
        # nomalize and pad for sam
        img_for_sam = self.preprocess(img).squeeze(0)

        if self.mode == 'val' and self.cfg.dataset.dino_as_input:
            prepare_for_dsam = self.generate_dino_list(img_path, instance, K, before_pad_size, original_size, raw_image, dataset_name)
        else:
            # generate data for object detection
            if 'obj_list' in instance.keys():
                prepare_for_dsam = self.generate_obj_list(instance, K, before_pad_size, original_size, raw_image, dataset_name)
                
                # random choose another frame
                if len(prepare_for_dsam) == 0:
                    print(img_path)
                    print('Warning: no valid object detected, return another sample')

                    prepare_for_dsam = []
            else:
                prepare_for_dsam = []
            
        if len(prepare_for_dsam) > self.cfg.dataset.max_dets:
            prepare_for_dsam = prepare_for_dsam[:self.cfg.dataset.max_dets]
        
        # calculate the vit pad size for depth head
        if self.cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // self.cfg.model.image_encoder.patch_size, before_pad_size[1] // self.cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, self.cfg.model.pad // self.cfg.model.image_encoder.patch_size)
        
        # padding depth
        depth_padded, depth_mask_padded = self.process_depth(img_for_sam, depth, before_pad_size, dataset_name)
        
        return_dict = {
            "images": img_for_sam,
            "masks": depth_mask_padded,
            'vit_pad_size': torch.tensor(vit_pad_size),
            "K": K.squeeze(0),
            "depth": depth_padded,
            "before_pad_size": torch.Tensor(before_pad_size),
            "prepare_for_dsam": prepare_for_dsam,
            "original_size": torch.tensor(original_size),

        }

        # post process image for dino, without padding
        img_for_dino = self.preprocess_dino(img).squeeze(0)

        return_dict.update({
            # input for dino
            "image_for_dino": img_for_dino,})

        return return_dict

    def __len__(self):
        return self.idx_cum[-1]
    
    def generate_obj_list(self, instance, K, before_pad_size, original_size, raw_image, dataset_name):
        
        prepare_for_dsam = []
        for obj in instance['obj_list']:
                
            # calculate the projcted 2d bbox
            x, y, z, w, h, l, yaw = obj['3d_bbox']
            pose = None
            if self.cfg.output_rotation_matrix:
                pose = obj['rotation_pose']
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, pose)
            if '2d_bbox_proj' in obj.keys() and obj['2d_bbox_proj'] != [-1, -1, -1, -1] and not self.cfg.add_cubercnn_for_ap_inference:
                bbox_2d_tensor = torch.tensor(obj['2d_bbox_proj'], dtype=torch.int)
                if self.cfg.dataset[self.mode][dataset_name].get("xywl_mode", False):
                    bbox_2d_tensor[2] += bbox_2d_tensor[0]
                    bbox_2d_tensor[3] += bbox_2d_tensor[1]
                bbox_2d_tensor = self.sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).squeeze(0)
                bbox_2d_tensor[0::2] = torch.clamp(bbox_2d_tensor[0::2], min=0, max=before_pad_size[1])
                bbox_2d_tensor[1::2] = torch.clamp(bbox_2d_tensor[1::2], min=0, max=before_pad_size[0])
            else:
                vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
                polygon_from_2d_box = MultiPoint(vertices_2d).convex_hull  

                img_canvas = box(0, 0, before_pad_size[1], before_pad_size[0]) 
                if polygon_from_2d_box.intersects(img_canvas):  
                    img_intersection = polygon_from_2d_box.intersection(img_canvas)
                    intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
                    min_x = min(intersection_coords[:, 0])
                    min_y = min(intersection_coords[:, 1])
                    max_x = max(intersection_coords[:, 0])
                    max_y = max(intersection_coords[:, 1])
                else:
                    continue

                bbox_2d_polygon = [min_x, min_y, max_x, max_y]
                
                bbox_2d_tensor = torch.tensor(bbox_2d_polygon, dtype=torch.int)


            
            if not self.filter_objects(obj, bbox_2d_tensor, before_pad_size, K, dataset_name):
                continue  # Skip this object
            
            # calculate center 2d from 3d center
            center_2d = project_to_image(np.array([[x, y, z]]), K.squeeze(0)).squeeze(0)
            center_2d_tensor = torch.tensor(center_2d)

            # modify yaw to [-pi, pi]
            if yaw > np.pi:
                yaw = yaw - 2 * np.pi
            if yaw < -np.pi:
                yaw = yaw + 2 * np.pi
                
            bbox_3d = [x, y, z, w, h, l, yaw]
            bbox_3d_tensor = torch.tensor(bbox_3d)

            human_prompt_coord = np.array([int((bbox_2d_tensor[0] + bbox_2d_tensor[2]) / 2), int((bbox_2d_tensor[1] + bbox_2d_tensor[3]) / 2)]) #* 0.5
            point_coords_tensor = torch.tensor(human_prompt_coord, dtype=torch.int).unsqueeze(0)
        
            if self.cfg.dataset.hack_point_prompt:
                point_coords_tensor = torch.tensor(self.cfg.dataset.hack_point_prompt).unsqueeze(0)
            if self.cfg.dataset.hack_box_prompt:
                bbox_2d_tensor = torch.tensor(self.cfg.dataset.hack_box_prompt)
            
            if self.cfg.dataset.perturbation_point_prompt and self.mode == 'train':
                point_coords_tensor = self.add_bbox_related_perturbations(point_coords_tensor, bbox_2d_tensor, perturbation_factor=self.cfg.dataset.perturbation_factor, num_pertuerbated_points = self.cfg.dataset.num_point_prompts)

            if self.cfg.dataset.perturbation_box_prompt and self.mode == 'train':
                box_coords = self.apply_bbox_perturbation(bbox_2d_tensor, before_pad_size)
            else:
                box_coords = bbox_2d_tensor.clone()
            
            if self.cfg.dataset.generate_point_prompts_via_mask:
                instance_id = dataset_name + "_" + str(obj['instance_id'])
                mask_path = f'exps/masks/{self.mode}/{instance_id}_mask.jpg'

                if os.path.exists(mask_path):
                    obj_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    point_coords_tensor = self.get_point_coords_from_mask(mask_path, num_point_prompts = self.cfg.dataset.num_point_prompts, original_point_coord_tensor = point_coords_tensor, bbox = bbox_2d_tensor)
            
            
            tmp_point_coords_list = [point_coords_tensor[i:i+1, ...] for i in range(point_coords_tensor.shape[0])]        
            
            for coord in tmp_point_coords_list:
    
                todo_dict = {
                    "bbox_2d": bbox_2d_tensor,
                    "point_coords": coord, #.unsqueeze(0),
                    "boxes_coords": box_coords,
                    "bbox_3d": bbox_3d_tensor.to(torch.float32),
                    "center_2d": center_2d_tensor.to(torch.float32),
                    "instance_id": obj.get('instance_id', None),
                    "depth_coords": torch.tensor(np.log(z)).to(torch.float32),
                    }

                if self.cfg.output_rotation_matrix:
                    todo_dict['rotation_pose'] = torch.tensor(obj['rotation_pose']).to(torch.float32)
                if self.cfg.add_cubercnn_for_ap_inference:
                    todo_dict['label'] = obj['label']
                    todo_dict['score'] = obj['score']
                    todo_dict['image_id'] = obj['image_id']
                    
                prepare_for_dsam.append(
                    todo_dict
                )
                
            # visualization code
           
            # [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = box_coords
            # coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2))]
            # to_draw = raw_image.permute(1, 2, 0).type(torch.uint8).numpy()
            # to_draw = cv2.cvtColor(to_draw, cv2.COLOR_RGB2BGR)
            # cv2.circle(to_draw, (int(point_coords_tensor[0][0]),int(point_coords_tensor[0][1])), 2, (0, 0, 255), 4)
            # cv2.circle(to_draw, (int(center_2d_tensor[0]),int(center_2d_tensor[1])), 2, (255, 255, 0), 4)
            # cv2.rectangle(to_draw, coor[0], coor[1], (0, 0, 255), 2)
            # cv2.imwrite('img_with_point_prompt.jpg', to_draw)

            # x, y, z, w, h, l, yaw = bbox_3d_tensor
            # vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, pose)
            # vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            # fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
            # to_draw = raw_image.permute(1, 2, 0).type(torch.uint8).numpy()
            # to_draw = cv2.cvtColor(to_draw, cv2.COLOR_RGB2BGR)
            # draw_bbox_2d(to_draw, vertices_2d)
            # cv2.circle(to_draw, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255) , 1)
            # cv2.imwrite('3D_test_change_K.png', to_draw)
            # import ipdb; ipdb.set_trace()
            # print('stop here')

        return prepare_for_dsam

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.cfg.model.pad - h
        padw = self.cfg.model.pad - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def preprocess_dino(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = x / 255
        IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        x = (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD

        return x
    
    def crop_hw(self, img, depth, K = None):
        
        if img.dim() == 4:
            img = img.squeeze(0)
        h, w = img.shape[1:3]  # 假设形状为 [C, H, W]
        assert max(h, w) % 112 == 0, "target_size must be divisible by 112"

        # 计算裁剪后尺寸，确保可以被 14 整除
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        # 计算裁剪区域的中心
        center_h, center_w = h // 2, w // 2

        # 计算裁剪的起始和结束索引
        start_h = center_h - new_h // 2
        start_w = center_w - new_w // 2

        # 按照中心裁剪图像和深度图
        img_cropped = img[:, start_h:start_h + new_h, start_w:start_w + new_w]
        depth_cropped = depth[start_h:start_h + new_h, start_w:start_w + new_w]

        K_cropped = None
        # 更新相机内参 K
        if K is not None:
            K_cropped = K.clone()  # 假设 K 是一个 numpy 数组
            K_cropped[0, 0, 2] -= (start_w)  # 更新 x 坐标
            K_cropped[0, 1, 2] -= (start_h)  # 更新 y 坐标

        return img_cropped.unsqueeze(0), depth_cropped, K_cropped

    def get_point_coords_from_mask(self, mask_path, num_point_prompts, original_point_coord_tensor, bbox, min_area=50, edge_margin=5):
        obj_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
        
        valid_regions = []
        largest_area = 0
        largest_region_label = -1
        
        # Loop through the regions to find the largest one
        for i in range(1, num_labels):  # skip background (label=0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                valid_regions.append(i)
                if area > largest_area:
                    largest_area = area
                    largest_region_label = i

        if largest_region_label == -1:
            print("No valid regions found.", mask_path)
            return original_point_coord_tensor
        
        # Get all the points in the largest region
        points = np.column_stack(np.where(labels == largest_region_label))
        
        # # Filter points to avoid edge areas
        # safe_points = []
        # for point in points:
        #     # Avoid points near the edges by checking if the point is within the "safe" region
        #     x, y = point
        #     if (x > edge_margin and x < obj_mask.shape[0] - edge_margin and
        #         y > edge_margin and y < obj_mask.shape[1] - edge_margin):
        #         # Only keep points inside the bbox
        #         xmin, ymin, xmax, ymax = bbox
        #         if xmin <= y <= xmax and ymin <= x <= ymax:
        #             safe_points.append(point)
        for i, point in enumerate(points):
            points[i] = point[::-1]  # Reverse (y, x) to (x, y)

        if len(points) == 0:
            print("No valid points found inside bbox.", mask_path)
            return original_point_coord_tensor
        
        # Calculate the bbox center
        xmin, ymin, xmax, ymax = bbox
        bbox_center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        
        # Compute distance to center for each point
        distances = np.linalg.norm(np.array(points) - bbox_center, axis=1)
        
        # Convert distances to a probability distribution (closer points have higher probability)
        probabilities = np.exp(-distances)  # Inverse exponential distance to give higher probability to closer points
        probabilities /= probabilities.sum()  # Normalize the probabilities

        # Randomly select points based on the computed probabilities
        mask_points_list = []
        if self.mode != 'train':
            num_point_prompts = 1

        for i in range(num_point_prompts):
            selected_point = points[np.random.choice(len(points), p=probabilities)]
            
            mask_points_list.append(selected_point)
        
        mask_points_list = np.array(mask_points_list)

        return torch.tensor(mask_points_list)
    
    def process_depth(self, img_for_sam, depth, before_pad_size, dataset_name):
        """
        Process the depth image with padding and apply depth mask based on dataset settings.

        Args:
            img_for_sam (Tensor): The image tensor for SAM (used for matching size).
            depth (Tensor): The depth image to be padded and processed.
            before_pad_size (tuple): The size of the depth image before padding.
            dataset_name (str): The name of the dataset to retrieve specific configuration.

        Returns:
            depth_padded (Tensor): The padded depth image.
            depth_mask_padded (Tensor): The padded depth mask indicating valid depths.
        """
        # Initialize padded depth and depth mask
        depth_padded = torch.zeros_like(img_for_sam[0])
        depth_mask_padded = torch.zeros_like(depth_padded)

        # Pad depth and mask
        depth_padded[:before_pad_size[0], :before_pad_size[1]] = depth
        depth_mask_padded[:before_pad_size[0], :before_pad_size[1]] = 1

        # Apply max/min distance filtering based on mode
        if self.dataset_dict[dataset_name].max_distance:
            depth_mask_padded[depth_padded >= self.dataset_dict[dataset_name].max_distance] = 0
        if self.dataset_dict[dataset_name].min_distance:
            depth_mask_padded[depth_padded <= self.dataset_dict[dataset_name].min_distance] = 0
        
        return depth_padded, depth_mask_padded

    def filter_objects(self, obj, bbox_2d_tensor, before_pad_size, K, dataset_name, min_size=5, min_height_ratio=0.05, min_depth=0.01, max_depth=100):
        """
        Filter out objects based on their visibility, truncation, center 2D bounding box, height, and depth.
        Args:
            obj (dict): A dictionary containing object information.
            before_pad_size (tuple): The original image size (height, width) before padding.
            K (torch.Tensor): The camera intrinsic matrix.
            min_height (int): Minimum height of the bounding box to keep the object.
            min_depth (float): Minimum depth value to keep the object.
            max_depth (float): Maximum depth value to keep the object.
        Returns:
            bool: True if the object passes the filter, False otherwise.
        """
        # import ipdb;ipdb.set_trace()
        if bbox_2d_tensor[2] - bbox_2d_tensor[0] < min_size or bbox_2d_tensor[3] - bbox_2d_tensor[1] < min_size:
            # print('a potential risk of bbox size')
            return False
        if bbox_2d_tensor[3] - bbox_2d_tensor[1] < min_height_ratio * before_pad_size[0]:
            # print('a potential risk of bbox height')
            return False
        
        # Get 3D bounding box parameters
        x, y, z, w, h, l, yaw = obj['3d_bbox']
        
        # Check if visibility and truncation are acceptable
        visibility = obj.get('visibility', 1)  # Default to 1 if not provided
        truncation = obj.get('truncation', 0)  # Default to 0 if not provided

        if visibility == -1:
            visibility = 1  # Default to fully visible if not provided
        if truncation == -1:
            truncation = 0  # Default to no truncation if not provided
        
        # Filter based on visibility and truncation
        if visibility < 0.333 or truncation > 0.33:
            # print('a potential risk of visibility or truncation')
            return False
        
        # Project the 3D center to 2D to get the center's location
        center_2d = project_to_image(np.array([[x, y, z]]), K.squeeze(0)).squeeze(0)
        
        # Apply filter on the 2D center: make sure it’s within the image boundaries
        if center_2d[0] < 0 or center_2d[0] > before_pad_size[1] or center_2d[1] < 0 or center_2d[1] > before_pad_size[0]:
            # print('a potential risk of center location')
            return False
        
        # Extract depth information: we will use the z-coordinate of the 3D center as depth
        depth = z
        
        # Check if depth is within the valid range
        if depth < min_depth or depth > max_depth:
            # print('a potential risk of depth')
            return False

        if not 'kitti' in dataset_name and not 'sunrgbd' in dataset_name and not 'arkitscenes' in dataset_name and not 'objectron' in dataset_name and not 'hypersim' in dataset_name and not 'nuscenes' in dataset_name:
            return True

        if self.cfg.dataset.zero_shot:
            # cubercnn categories
            thing_dataset_id_to_contiguous_id =  {"0": 0, "1": 1, "3": 2, "4": 3, "5": 4, "8": 5, "9": 6, "10": 7, "11": 8, "12": 9, "13": 10, "14": 11, "15": 12, "16": 13, "17": 14, "18": 15, "19": 16, "20": 17, "21": 18, "22": 19, "23": 20, "24": 21, "25": 22, "26": 23, "27": 24, "28": 25, "29": 26, "30": 27, "31": 28, "32": 29, "33": 30, "34": 31, "35": 32, "36": 33, "37": 34, "38": 35, "39": 36, "40": 37, "42": 38, "43": 39, "44": 40, "45": 41, "46": 42, "47": 43, "48": 44, "49": 45, "52": 46, "53": 47, "57": 48, "61": 49}
            if f"{obj['label']}" not in thing_dataset_id_to_contiguous_id.keys():
                if self.mode == 'train':
                    return False
                elif self.cfg.inference_basic:
                    return False
        #     else:
        #         if self.mode != 'train' and self.cfg.inference_novel:
        #             return False
        # else:
        #     thing_dataset_id_to_contiguous_id = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, "21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29, "30": 30, "31": 31, "32": 32, "33": 33, "34": 34, "35": 35, "36": 36, "37": 37, "38": 38, "39": 39, "40": 40, "41": 41, "42": 42, "43": 43, "44": 44, "45": 45, "46": 46, "47": 47, "48": 48, "49": 49, "50": 50, "51": 51, "52": 52, "53": 53, "54": 54, "55": 55, "56": 56, "57": 57, "58": 58, "59": 59, "60": 60, "61": 61, "62": 62, "63": 63, "64": 64, "65": 65, "66": 66, "67": 67, "68": 68, "69": 69, "70": 70, "71": 71, "72": 72, "73": 73, "74": 74, "75": 75, "76": 76, "77": 77, "78": 78, "79": 79, "80": 80, "81": 81, "82": 82, "83": 83, "84": 84, "85": 85, "86": 86, "87": 87, "88": 88, "89": 89, "90": 90, "91": 91, "92": 92, "94": 93, "95": 94, "96": 95, "97": 96}
        #     if f"{obj['label']}" not in thing_dataset_id_to_contiguous_id.keys():
        #         return False
        # If all checks pass, the object is valid

        return True

    def apply_bbox_perturbation(self, bbox_2d_tensor, before_pad_size, max_perturbation_factor=0.05):
        """
        Apply random perturbations to the bounding box coordinates.
        
        Args:
            bbox_2d_tensor (Tensor): A tensor of shape [4] representing the bounding box coordinates 
                                    in the format [x1, y1, x2, y2].
            max_perturbation_factor (float): The maximum perturbation factor as a percentage of the bbox size.
        
        Returns:
            Tensor: A tensor representing the perturbed bounding box coordinates.
        """
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = bbox_2d_tensor
        
        # Calculate width and height of the bbox
        width = x2 - x1
        height = y2 - y1
        
        # Apply random perturbation within the specified factor range
        perturbation_x1 = random.uniform(-max_perturbation_factor, max_perturbation_factor) * width
        perturbation_y1 = random.uniform(-max_perturbation_factor, max_perturbation_factor) * height
        perturbation_x2 = random.uniform(-max_perturbation_factor, max_perturbation_factor) * width
        perturbation_y2 = random.uniform(-max_perturbation_factor, max_perturbation_factor) * height
        
        # Apply perturbation to each corner of the bbox
        x1_perturbed = x1 + perturbation_x1
        y1_perturbed = y1 + perturbation_y1
        x2_perturbed = x2 + perturbation_x2
        y2_perturbed = y2 + perturbation_y2
        
        # Ensure that the new bounding box still maintains its validity (x1 < x2, y1 < y2)
        x1_perturbed = max(0, min(x1_perturbed, x2_perturbed))  # Prevent x1 from going out of bounds
        y1_perturbed = max(0, min(y1_perturbed, y2_perturbed))  # Prevent y1 from going out of bounds
        x2_perturbed = max(x1_perturbed, min(x2_perturbed, before_pad_size[1]))  # Prevent x2 from going out of bounds
        y2_perturbed = max(y1_perturbed, min(y2_perturbed, before_pad_size[0]))  # Prevent y2 from going out of bounds
        
        # Return the perturbed bbox
        return torch.tensor([x1_perturbed, y1_perturbed, x2_perturbed, y2_perturbed])

    def add_bbox_related_perturbations(self, point_coords_tensor, bbox_2d_tensor, perturbation_factor=0.05, num_pertuerbated_points = 1):
        """
        Add perturbations to the input coordinates and bounding boxes based on the size of the 2D bounding boxes.
        
        Args:
        - input_dict (dict): A dictionary containing 'point_coords' (and optionally 'boxes_coords').
        - bbox_2d_tensor (Tensor): The ground truth 2D bounding boxes (x_min, y_min, x_max, y_max).
        - perturbation_factor (float): The factor that controls the size of the perturbation. Default is 0.05.
        - device_id (optional): The device to transfer the tensors to (e.g., 'cuda', 'cpu').
        
        Returns:
        - input_dict (dict): The updated dictionary with perturbed coordinates.
        """
        # Ensure bbox dimensions are of float type (this avoids the error with `torch.randn_like`)
        # import ipdb; ipdb.set_trace()
        bbox_2d_tensor = bbox_2d_tensor.to(point_coords_tensor.dtype)

        # Calculate the width and height of 2D bounding boxes
        bbox_width = bbox_2d_tensor[2] - bbox_2d_tensor[0]
        bbox_height = bbox_2d_tensor[3] - bbox_2d_tensor[1]

        bbox_width = bbox_width.repeat(num_pertuerbated_points)
        bbox_height = bbox_height.repeat(num_pertuerbated_points)
        
        # Generate random perturbations based on bbox dimensions (ensure they are float)
        perturbation_x = torch.randn_like(bbox_width) * bbox_width * perturbation_factor
        perturbation_y = torch.randn_like(bbox_height) * bbox_height * perturbation_factor
        point_coords_tensor = point_coords_tensor.repeat(num_pertuerbated_points, 1)
        # Add the perturbations to the point coordinates
        perturbed_point_coords = point_coords_tensor + torch.stack([perturbation_x, perturbation_y], dim=-1)

        return perturbed_point_coords

    def generate_dino_list(self, img_path, instance, K, before_pad_size, original_size, raw_image, dataset_name):
        TEXT_PROMPT = ''
        if 'kitti' in dataset_name:
            if self.cfg.inference_basic:
                TEXT_PROMPT += "pedestrian . car . cyclist . van . truck . "
            if self.cfg.inference_novel:
                TEXT_PROMPT += "tram . "
        if 'nuscenes' in dataset_name:
            if self.cfg.inference_basic:
                TEXT_PROMPT += "pedestrian . car . truck . traffic cone . barrier . motorcycle . bicycle . bus . trailer . "
            if self.cfg.inference_novel:
                TEXT_PROMPT += " . "
        if 'arkitscenes' in dataset_name:
            if self.cfg.inference_basic:
                TEXT_PROMPT += "table . bed . sofa . television . refrigerator . chair . oven . machine . stove . shelves . sink . cabinet . bathtub . toilet . "
            if self.cfg.inference_novel:
                TEXT_PROMPT += "fireplace . "
        if 'sunrgbd' in dataset_name:
            if self.cfg.inference_basic:
                TEXT_PROMPT += "bicycle . books . bottle . chair . cup . laptop . shoes . towel . blinds . window . lamp . shelves . mirror . sink . cabinet . bathtub . door . toilet . desk . box . bookcase . picture . table . counter . bed . night stand . pillow . sofa . television . floor mat . curtain . clothes . stationery . refrigerator . bin . stove . oven . machine . "
            if self.cfg.inference_novel:
                TEXT_PROMPT += "monitor . bag . dresser . board . printer . keyboard . painting . drawers . microwave . computer . kitchen pan . potted plant . tissues . rack . tray . toys . phone . podium . cart . soundsystem . "
        if 'objectron' in dataset_name:
            if self.cfg.inference_basic:
                TEXT_PROMPT += "bicycle . books . bottle . camera . cereal box . chair . cup . laptop . shoes . "
            if self.cfg.inference_novel:
                TEXT_PROMPT += " . "
        if 'hypersim' in dataset_name:
            if self.cfg.inference_basic:
                TEXT_PROMPT += "books . chair . towel . blinds . window . lamp . shelves . mirror . sink . cabinet . bathtub . door . desk . box . bookcase . picture . table . counter . bed . night stand . pillow . sofa . television . floor mat . curtain . clothes . stationery . refrigerator . "
            if self.cfg.inference_novel:
                TEXT_PROMPT += " . "
        if '3rscan' in dataset_name: 
            if self.cfg.inference_basic:
                TEXT_PROMPT += "chair . box . sink . stove . curtain . door . table . window . pillow . sofa . cabinet . lamp . picture . bed . desk . toilet . counter . oven . clothes . shoes . bottle . towel . laptop . bin . mirror . cup . bathtub . blinds . machine . books . refrigerator . "
            if self.cfg.inference_novel:
                TEXT_PROMPT += "trash can . table lamp . kitchen cabinet . fridge . object . rack . dining table . soap dish . kettle . toaster . wall . pipe . hood . sponge . floor . shoe . heater . oven glove . radiator . menu . frame . commode . blanket . tv . plate . coffee table . fan . tv stand . decoration . plant . stool . treadmill . shelf . item . showcase . couch . fireplace . pile of candles . board . ceiling . wardrobe . nightstand . light . bag . ventilator . printer . vase . flowers . ottoman . doorframe . shower door . bidet . toilet paper . shower wall . shower floor . bath cabinet . clothes dryer . garbage bin . hand dryer . cupboard . side table . armchair . gymnastic ball . clutter . firewood box . bucket . drum . tree decoration . rocking chair . couch table . bench . jar . windowsill . garbage . kitchen counter . telephone . cutting board . paper towel . basket . shoe rack . kitchen object . coffee machine . container . microwave . backpack . suitcase . toilet brush . toilet paper dispenser . tube . socket . tennis raquet . stand . pile of papers . pack . monitor . cap . ball . pc . folder . pile of books . shower curtain . clock . kitchen hood . computer desk . armoire . storage bin . cleanser . carpet . chest . pot . stuffed animal . scale . pile of bottles . basin . bar . kitchen appliance . food . napkins . book . wood . stair . ladder . objects . trashcan . magazine rack . fence . bedside table . jacket . papers . organizer . drawer . flower . grass . rag . kitchen sofa . fruit plate . player . round table . sofa chair . cushion . screen . laundry basket . wall /other room . floor /other room . shower . recycle bin . washing machine . drying machine . cart . washing powder . bookshelf . glass . linen . bread . storage container . pan . extractor fan . closet . footstool . dining chair . "
        if 'cityscapes' in dataset_name:
            if self.cfg.inference_basic:
                TEXT_PROMPT += "trailer . car . truck . bus . bicycle . motorcycle . "
            if self.cfg.inference_novel:
                TEXT_PROMPT += "tram . "
        if 'waymo' in dataset_name:
            if self.cfg.inference_basic:
                TEXT_PROMPT += "pedestrian . car . cyclist . "
        if 'kit_leaderboard' in dataset_name:
            TEXT_PROMPT += "car . "
        # import ipdb; ipdb.set_trace()

        check_label = TEXT_PROMPT.split(' . ')
        image_source, image = load_image(img_path)
        # import ipdb;ipdb.set_trace()
        if self.cfg.dataset.previous_metric:
            target_str = TEXT_PROMPT
        else:
            target_str = ''
            for obj in instance['obj_list']:
                label_id = obj['label']
                temp_label = self.category_id['thing_classes'][label_id]
                if temp_label in check_label and temp_label not in target_str:
                    target_str += f"{temp_label} . "

        # print(target_str)
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image,
            caption=target_str,
            box_threshold=self.BOX_TRESHOLD,
            text_threshold=self.TEXT_TRESHOLD,
            remove_combined=True,
        )
        # print(phrases)
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    
        # import ipdb;ipdb.set_trace()
        prepare_for_dsam = []
        image_id = None
        if len(instance['obj_list']) > 0:
            image_id = instance['obj_list'][0]['image_id']
        for i, box in enumerate(xyxy):
            if phrases[i] == '' or phrases[i] not in check_label:
                continue
            
            bbox_2d_tensor = box
            bbox_2d_tensor = self.sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).squeeze(0)
            bbox_2d_tensor[0::2] = torch.clamp(bbox_2d_tensor[0::2], min=0, max=before_pad_size[1])
            bbox_2d_tensor[1::2] = torch.clamp(bbox_2d_tensor[1::2], min=0, max=before_pad_size[0])

            # import ipdb;ipdb.set_trace()
            # to_draw = raw_image.permute(1, 2, 0).type(torch.uint8).numpy()
            # to_draw = cv2.cvtColor(to_draw, cv2.COLOR_RGB2BGR)
            # [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = bbox_2d_tensor
            # coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2))]
           
            # cv2.rectangle(to_draw, coor[0], coor[1], (0, 0, 255), 2)
            # cv2.imwrite('img_with_point_prompt.jpg', to_draw)

            human_prompt_coord = np.array([int((bbox_2d_tensor[0] + bbox_2d_tensor[2]) / 2), int((bbox_2d_tensor[1] + bbox_2d_tensor[3]) / 2)]) #* 0.5
            point_coords_tensor = torch.tensor(human_prompt_coord, dtype=torch.int).unsqueeze(0)
            box_coords = bbox_2d_tensor.clone()
            def expand_box(box_coords: torch.Tensor, scale: float = 0.1, image_size: tuple = None) -> torch.Tensor:
                """
                Expand a bounding box by a scale factor of its width and height.

                Args:
                    box_coords (torch.Tensor): Tensor of shape [4] with format [x1, y1, x2, y2].
                    scale (float): Percentage of width/height to expand (e.g., 0.1 for 10%).
                    image_size (tuple, optional): (H, W) to clamp the box within image boundaries.

                Returns:
                    torch.Tensor: Expanded bounding box of shape [4], type torch.int.
                """
                x1, y1, x2, y2 = box_coords.float()
                w = x2 - x1
                h = y2 - y1
                dx = w * scale
                dy = h * scale

                x1_exp = x1 - dx
                y1_exp = y1 - dy
                x2_exp = x2 + dx
                y2_exp = y2 + dy

                expanded_box = torch.tensor([x1_exp, y1_exp, x2_exp, y2_exp])

                if image_size is not None:
                    H, W = image_size
                    expanded_box[0::2] = torch.clamp(expanded_box[0::2], 0, W)
                    expanded_box[1::2] = torch.clamp(expanded_box[1::2], 0, H)

                return expanded_box.round().to(torch.int)

            # 扩展后的 box（假设原图尺寸为 before_pad_size）
            box_coords = expand_box(box_coords, scale=0.1, image_size=before_pad_size)

            todo_dict = {
                "bbox_2d": bbox_2d_tensor,
                "point_coords": point_coords_tensor,
                "boxes_coords": box_coords,
                "bbox_3d": torch.tensor([-1, -1, -1, -1, -1, -1, -1]),
                "center_2d": torch.tensor([-1, -1]),
                "instance_id": 'tbd',
                'name': phrases[i],
                # "instance_mask": seg_mask,
                }
            if self.cfg.output_rotation_matrix:
                todo_dict['rotation_pose'] = torch.eye(3)
            
            if self.cfg.add_cubercnn_for_ap_inference:
                label = phrases[i]
                
                if label not in self.category_id['thing_classes']:
                    # import ipdb;ipdb.set_trace()
                    continue
                todo_dict['label'] = self.category_id['thing_classes'].index(label)
                todo_dict['score'] = logits[i].item()
                todo_dict['image_id'] = image_id
            
            prepare_for_dsam.append(
                todo_dict
            )

        return prepare_for_dsam
