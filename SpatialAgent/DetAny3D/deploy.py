from train_utils import *
from wrap_model import WrapModel

from PIL import Image
import cv2
import yaml
import os
import torch.nn as nn
import torch.distributed as dist
from box import Box
import random


import gradio as gr
from gradio_image_prompter import ImagePrompter

from groundingdino.util.inference import load_model
from groundingdino.util.inference import predict as dino_predict

from torchvision.ops import box_convert
import colorsys
import json
import hashlib
import io

with open('./detect_anything/configs/demo.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
cfg = Box(cfg)

def generate_image_token(image: Image.Image) -> str:
    """根据 PIL.Image 生成唯一 Token（SHA-256 哈希值）"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # 转换为 PIL.Image
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')  # 统一格式，避免存储差异
    return hashlib.sha256(img_bytes.getvalue()).hexdigest()

# 禁用分布式初始化
torch.distributed.is_available = lambda: False
torch.distributed.is_initialized = lambda: False
torch.distributed.get_world_size = lambda group=None: 1  # 总是返回 1，表示单进程模式
torch.distributed.get_rank = lambda group=None: 0  # 总是返回 rank 0

my_sam_model = WrapModel(cfg)
checkpoint = torch.load(cfg.resume, map_location=f'cuda:0')
new_model_dict = my_sam_model.state_dict()
for k,v in new_model_dict.items():
    if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
        new_model_dict[k] = checkpoint['state_dict'][k].detach()
my_sam_model.load_state_dict(new_model_dict)
my_sam_model.to('cuda:0')
my_sam_model.setup()
my_sam_model.eval()
sam_trans = ResizeLongestSide(cfg.model.pad)

dino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
dino_model.eval()
BOX_TRESHOLD = 0.37
TEXT_TRESHOLD = 0.25

import groundingdino.datasets.transforms as T
def convert_image(img):

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(img, 'RGB')
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

def crop_hw(img):
        
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
    
    return img_cropped.unsqueeze(0)

def preprocess(x, cfg):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    sam_pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
    sam_pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
    x = (x - sam_pixel_mean) / sam_pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = cfg.model.pad - h
    padw = cfg.model.pad - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def preprocess_dino(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = x / 255
    IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    x = (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD

    return x

def adjust_brightness(color, factor=1.5, v_min=0.3):
    """在 HSV 空间调整亮度，避免过暗"""
    r, g, b = color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = max(v, v_min) * factor  # 强制最低亮度 + 增强
    v = min(v, 1.0)  # 防止过曝
    return colorsys.hsv_to_rgb(h, s, v)

def draw_text(im, text, pos, scale=0.4, color='auto', font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255),
              blend=0.33, lineType=1):

    text = str(text)
    pos = [int(pos[0]), int(pos[1])]

    if color == 'auto':
        
        if bg_color is not None:
            color = (0, 0, 0) if ((bg_color[0] + bg_color[1] + bg_color[2])/3) > 127.5 else (255, 255, 255)
        else:
            color = (0, 0, 0) 

    if bg_color is not None:

        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(x_s + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)
        
        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)
def predict(input_dict, text):
    return input_dict

    with torch.no_grad():
        img, points = input_dict['image'], input_dict['points']
        image_token = generate_image_token(img)

        pixels = np.array(img).reshape(-1, 3) / 255.0

        # 改进点1：根据亮度加权采样（避免全随机选到过多暗色）
        brightness = pixels.mean(axis=1)  # 计算每个像素的亮度
        prob = brightness / brightness.sum()  # 亮度越高采样概率越大
        sampled_indices = np.random.choice(pixels.shape[0], 100, p=prob, replace=False)
        sampled_colors = pixels[sampled_indices]
        # 改进点2：按亮度排序而非直接排序
        sampled_colors = sorted(sampled_colors, key=lambda c: colorsys.rgb_to_hsv(*c)[2])
        # 应用亮度增强
        adjusted_colors = [adjust_brightness(c, factor=2.0, v_min=0.4) for c in sampled_colors]

        if img is None:
            return "No image received"
        
        label_list = []
        bbox_2d_list = []
        point_coords_list = []
        for point in points:
            if point[2] == 1  or point[5] == 1:
                # TODO point coords
                human_prompt_coord = np.array([point[0], point[1]]) #* 0.5
                point_coords_tensor = torch.tensor(human_prompt_coord, dtype=torch.int).unsqueeze(0)
                point_coords_list.append(point_coords_tensor)
                
                
            if point[2] == 2 and point[5] == 3:
                x1, y1 = point[:2]
                x2, y2 = point[3:5]
                bbox_2d = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                bbox_2d_list.append(bbox_2d)
                label_list.append("Unknow")
        
        if len(bbox_2d_list) > 0 and len(point_coords_list) > 0:
            raise gr.Error("Can not hadle bounding box and point at the same time.")
        
        if len(point_coords_list) == 0:
            mode = 'box'
        
        else:
            mode = 'point'
            label_list = ["Unknow"]

        image_source_dino, image_dino = convert_image(img)
        if text != '' and mode == 'point':
            gr.Warning("Both text and point prompt input, follow the point prompt")
            
        boxes, logits, phrases = dino_predict(
            model=dino_model,
            image=image_dino,
            caption=text,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            remove_combined=False,
        )
        h, w, _ = image_source_dino.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        for i, box in enumerate(xyxy):
            if mode == 'box':
                bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
                label_list.append(phrases[i])
            elif mode == 'point':
                pass
        

        if len(bbox_2d_list) == 0 and len(point_coords_list) == 0:
            raise gr.Error("No objects found in the image. Please try again with a different prompt.")
            
        raw_img = img.copy()
        original_size = tuple(img.shape[:-1])
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        img = img.unsqueeze(0)

        # Apply your model's transformations
        img = sam_trans.apply_image_torch(img)
        img = crop_hw(img)
        before_pad_size = tuple(img.shape[2:])
        
        img_for_sam = preprocess(img, cfg).to('cuda:0')
        img_for_dino = preprocess_dino(img).to('cuda:0')

        resize_ratio = max(img_for_sam.shape) / max(original_size)
        image_h, image_w = int(before_pad_size[0]), int(before_pad_size[1])

        if cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // cfg.model.image_encoder.patch_size, before_pad_size[1] // cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (cfg.model.pad // cfg.model.image_encoder.patch_size, cfg.model.pad // cfg.model.image_encoder.patch_size)

        if mode == 'box':
            bbox_2d_tensor = torch.tensor(bbox_2d_list)
            bbox_2d_tensor = sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to('cuda:0')
            print(bbox_2d_tensor.shape)
            input_dict = {
                "images": img_for_sam,
                'vit_pad_size': torch.tensor(vit_pad_size).to('cuda:0').unsqueeze(0),
                "images_shape": torch.Tensor(before_pad_size).to('cuda:0').unsqueeze(0),
                "image_for_dino": img_for_dino,
                "boxes_coords": bbox_2d_tensor,
            }
        if mode == 'point':

            points_2d_tensor = torch.stack(point_coords_list, dim=1).to('cuda:0')
            points_2d_tensor = sam_trans.apply_coords_torch(points_2d_tensor, original_size)
            input_dict = {
                "images": img_for_sam,
                'vit_pad_size': torch.tensor(vit_pad_size).to('cuda:0').unsqueeze(0),
                "images_shape": torch.Tensor(before_pad_size).to('cuda:0').unsqueeze(0),
                "image_for_dino": img_for_dino,
                "point_coords": points_2d_tensor,
            }


        ret_dict = my_sam_model(input_dict)

        K = ret_dict['pred_K']
        decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K)
        rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])
        pred_box_ious = ret_dict.get('pred_box_ious', None)

        origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_for_sam[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        todo = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        K = K.detach().cpu().numpy()
        
        for i in range(len(decoded_bboxes_pred_2d)):
            x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
            rot_mat_i = rot_mat[i].detach().cpu().numpy()
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
            vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
            color = adjusted_colors[i]

            color = [min(255, c*255) for c in color]
            
            best_j = torch.argmax(pred_box_ious[i])  # Get the index of the best IoU box
            iou_score = pred_box_ious[i][best_j].item()  # Get the IoU score as a scalar

            # Draw the 2D bounding box (predicted)
            draw_bbox_2d(todo, vertices_2d, color=(int(color[0]), int(color[1]), int(color[2])), thickness=3)
            if label_list[i] is not None:
                draw_text(todo, f"{label_list[i]} {[round(c, 2) for c in decoded_bboxes_pred_3d[i][3:6].detach().cpu().numpy().tolist()]}", box_cxcywh_to_xyxy(decoded_bboxes_pred_2d[i]).detach().cpu().numpy().tolist(), scale=0.50*todo.shape[0]/500, bg_color=color)
     
        cv2.imwrite(f'./exps/deploy/{image_token}.jpg', todo)
        todo = todo / 255
        todo = np.clip(todo, -1.0, 1.0)
        rgb_image = cv2.cvtColor(todo, cv2.COLOR_BGR2RGB)
        
    return rgb_image


iface = gr.Interface(
    predict,
    [ImagePrompter(show_label=False),
    gr.Textbox(label="Please enter the prompt, e.g. a person, seperate different prompt with ' . '")],
    outputs=[
        gr.Image(),  # 图像输出
    ]
)


# 启动 Gradio 应用
iface.launch(share=True, server_port=7861)
