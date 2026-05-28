from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from detect_anything.datasets import transforms_shir as transforms
import open3d as o3d
import torch.nn.functional as F
from PIL import Image
import cv2
import pickle
import numpy as np
from detect_anything.utils.transforms import ResizeLongestSide
from detect_anything.utils.amg import batched_mask_to_box
from detect_anything.mylogger import *
from copy import deepcopy
from einops import rearrange

import matplotlib

from utils import *

class TestDataset(Dataset):

    def __init__(self, 
                 cfg,
                 transform,
                 mode,
                 pixel_mean = [123.675, 116.28, 103.53],
                 pixel_std = [58.395, 57.12, 57.375],):

        self.transform = transform
        # import ipdb;ipdb.set_trace()
        if mode == 'train':
            # import ipdb;ipdb.set_trace()
            self.pkl_path = cfg.dataset.train.pkl_path
            with open(self.pkl_path, 'rb') as f:
                self.any_dataset_pkl = pickle.load(f)
        else:
            self.pkl_path = cfg.dataset.val.pkl_path
            with open(self.pkl_path, 'rb') as f:
                self.any_dataset_pkl = pickle.load(f)

        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.target_size = cfg.model.pad
        self.sam_trans = ResizeLongestSide(self.target_size)
        self.cfg = cfg
        self.mode = mode

    def __getitem__(self, index):
        # import ipdb;ipdb.set_trace()
        K = self.any_dataset_pkl[index]['K'].astype(np.float32)
        K = torch.tensor(K)
        
        depth_path = self.any_dataset_pkl[index]['depth_path']
        if[depth_path][-4:] == '.png':
            depth = np.array(Image.open(depth_path)).astype(np.float32)
            if self.mode == 'train':
                depth = depth / self.cfg.dataset.train[dataset_name].metric_scale
            else:
                depth = depth / self.cfg.dataset.val[dataset_name].metric_scale
        else:
            depth = np.load(depth_path).astype(np.float32)
        # depth = self.any_dataset_pkl[index]['depth']
        # import ipdb;ipdb.set_trace()
        img_path = self.any_dataset_pkl[index]['img_path']
        print('The original image path: ', img_path)
        todo_img = cv2.imread(img_path)
        todo_img = cv2.cvtColor(todo_img, cv2.COLOR_BGR2RGB)
        
        original_size = tuple(todo_img.shape[:-1])
        todo_img, depth = self.transform(todo_img, depth)
        
        cropped_size = tuple(todo_img.shape[1:3])
        K[0, 0, 2] = K[0, 0, 2] + (cropped_size[0] - original_size[0]) / 2
        K[0, 1, 2] = K[0, 1, 2] + (cropped_size[1] - original_size[1]) / 2

        todo_img = todo_img.unsqueeze(0)
        todo_img = self.sam_trans.apply_image_torch(todo_img)
        depth = self.sam_trans.apply_depth_torch(depth.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        # import ipdb;ipdb.set_trace()
        mask = torch.ones_like(depth)
        mask[depth == torch.inf] = 0

        before_pad_size = tuple(todo_img.shape[-2:])
        resize_ratio = before_pad_size[1] / cropped_size[1]
        K[0, 0] = K[0, 0] * resize_ratio
        K[0, 1] = K[0, 1] * resize_ratio
        vit_pad_size = (before_pad_size[0] // 16, before_pad_size[1] // 16)
        incid = intrinsic2incidence(K, 1, before_pad_size[0], before_pad_size[1], 'cpu').squeeze(0).squeeze(-1).permute(2,0,1)
        
        todo_img = self.preprocess(todo_img).squeeze(0)

        depth_padded = torch.zeros_like(todo_img[0])
        depth_mask_padded = torch.zeros_like(depth_padded)
        incid_padded = torch.zeros_like(todo_img)

        depth_padded[:before_pad_size[0], :before_pad_size[1]] = depth
        depth_mask_padded[:before_pad_size[0], :before_pad_size[1]] = mask
        incid_padded[:, :before_pad_size[0], :before_pad_size[1]] = incid
        incid_mask = torch.zeros_like(incid_padded)
        incid_mask[:, :before_pad_size[0], :before_pad_size[1]] = 1

        return {
            "images": todo_img,
            "masks": depth_mask_padded,
            'vit_pad_size': torch.tensor(vit_pad_size),
            "incid_mask": incid_mask,
            "K": K.squeeze(0),
            "depth": depth_padded,
            "incidence": incid_padded,
            "before_pad_size": torch.Tensor(before_pad_size),
        }
        
    def __len__(self):
        return len(self.any_dataset_pkl)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.target_size - h
        padw = self.target_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

from box import Box
import yaml

with open('segment_anything/configs/test_dataset_config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
import ipdb;ipdb.set_trace()
cfg = Box(cfg)

transforms_train, transforms_test = get_depth_transform()
test_data = TestDataset(cfg, transforms_test, 'val')
instance = test_data[20]
img = instance["images"]
origin_img = torch.Tensor(
    [58.395, 57.12, 57.375]).view(-1, 1, 1) * img.detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
todo = cv2.cvtColor(origin_img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
cv2.imwrite('segment_anything/datasets/test_samples/img.png', todo)

depth = instance["depth"]
mask = instance["masks"]
mask[depth > 20] = 0
# import ipdb;ipdb.set_trace()

max_depth = np.nanmax(depth[depth != np.inf])  # 获取有效深度的最大值
# 遍历深度图，绘制点云信息
for i in range(len(depth)):
    # import ipdb;ipdb.set_trace()
    for j in range(len(depth[i])):
        # 只有在深度有效且通过遮罩的情况下才绘制
        if depth[i, j] != np.inf and mask[i, j] != 0 and depth[i, j] != 0:
            color = int(depth[i, j] / 80 * 255)
            cv2.circle(todo, (j, i), radius=1, thickness=1, color=(255 - color, color, 0))
            # # 根据深度值调整颜色，深度越远颜色越接近红色，越近接近绿色
            depth_ratio = min(depth[i, j] / 20, 1.0)  # 确保深度比在[0, 1]之间
            color = (0, int(255 * (1 - depth_ratio)), int(255 * depth_ratio))  # 深度颜色映射

            # 在图像上绘制点
            cv2.circle(todo, (j, i), radius=3, thickness=-1, color=color)
            
cv2.imwrite('segment_anything/datasets/test_samples/draw_point_cloud_on_img.png', todo)

# depth[depth > 120] = 120
d = colorize(depth, 0, depth.max())
from PIL import Image
Image.fromarray(d).save("segment_anything/datasets/test_samples/depth.png")

incidence = instance['incidence'].numpy()
incidence_x = (incidence[0] + 1)/2 * 255
cv2.imwrite('segment_anything/datasets/test_samples/incidence_x.png', incidence_x)
incidence_y = (incidence[1] + 1)/2 * 255
cv2.imwrite('segment_anything/datasets/test_samples/incidence_y.png', incidence_y)
incidence_z = (incidence[2] + 1)/2 * 255
cv2.imwrite('segment_anything/datasets/test_samples/incidence_z.png', incidence_z)

origin_img = torch.Tensor(
    [58.395, 57.12, 57.375]).view(-1, 1, 1) * instance['images'].detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
color_image = origin_img.clone()[:, :int(instance['before_pad_size'][ 0]), :int(instance['before_pad_size'][ 1])].permute(1,2,0)
color_image = Image.fromarray(color_image.cpu().numpy().astype('uint8'), 'RGB')
import pdb;pdb.set_trace()
pred = depth[:int(instance['before_pad_size'][ 0]), :int(instance['before_pad_size'][ 1])].cpu().numpy()
FX = instance["K"][0,0].cpu().numpy()
FY = instance["K"][1,1].cpu().numpy()
BX = instance["K"][0,2].cpu().numpy()
BY = instance["K"][1,2].cpu().numpy()
x, y = np.meshgrid(np.arange(int(instance['before_pad_size'][ 1])), np.arange(int(instance['before_pad_size'][ 0])))
x = (x - BX) / FX
y = (y - BY) / FY
z = np.array(pred)
points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
colors = np.array(color_image).reshape(-1, 3) / 255.0
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(f"segment_anything/datasets/test_samples/visual_pc.ply", pcd)
print(instance['before_pad_size'])