# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import yaml
from box import Box
import os
import io
import sys
import random
import colorsys
import hashlib
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, predict as dino_predict
current_dir = os.path.dirname(os.path.abspath(__file__))
import groundingdino.datasets.transforms as T

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from train_utils import decode_bboxes, rotation_6d_to_matrix, box_cxcywh_to_xyxy, compute_3d_bbox_vertices, project_to_image

# Load configuration from YAML file
with open('./DetAny3D/detect_anything/configs/demo.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
cfg = Box(cfg)

# Initialize WrapModel
from wrap_model import WrapModel
my_sam_model = WrapModel(cfg)

# Load checkpoint for WrapModel
checkpoint = torch.load(cfg.resume, map_location='cuda:0')
new_model_dict = my_sam_model.state_dict()
for k, v in new_model_dict.items():
    if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
        new_model_dict[k] = checkpoint['state_dict'][k].detach()
my_sam_model.load_state_dict(new_model_dict)

# Move WrapModel to GPU and set to evaluation mode
my_sam_model.to('cuda:0')
my_sam_model.setup()
my_sam_model.eval()

# Initialize GroundingDINO model
dino_model = load_model(
    "./DetAny3D/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    "./DetAny3D/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
)
dino_model.eval()

# Define thresholds for GroundingDINO
BOX_TRESHOLD = 0.37
TEXT_TRESHOLD = 0.25

# Initialize image transformation for WrapModel
from train_utils import ResizeLongestSide
sam_trans = ResizeLongestSide(cfg.model.pad)


# Define Image Processing Functions

def convert_image(img):
    """
    Convert an image to a format suitable for processing.
    Applies resizing, normalization, and transformation.
    """
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
    """
    Crop the image to ensure dimensions are divisible by 14.
    """
    if img.dim() == 4:
        img = img.squeeze(0)
    h, w = img.shape[1:3]
    assert max(h, w) % 112 == 0, "target_size must be divisible by 112"

    new_h = (h // 14) * 14
    new_w = (w // 14) * 14

    center_h, center_w = h // 2, w // 2
    start_h = center_h - new_h // 2
    start_w = center_w - new_w // 2

    img_cropped = img[:, start_h:start_h + new_h, start_w:start_w + new_w]
    return img_cropped.unsqueeze(0)


def preprocess(x, cfg):
    """
    Normalize pixel values and pad the image to a square input.
    """
    sam_pixel_mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
    sam_pixel_std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
    x = (x - sam_pixel_mean) / sam_pixel_std

    h, w = x.shape[-2:]
    padh = cfg.model.pad - h
    padw = cfg.model.pad - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def preprocess_dino(x):
    """
    Normalize pixel values for GroundingDINO model.
    """
    x = x / 255
    IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    x = (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD
    return x


def adjust_brightness(color, factor=1.5, v_min=0.3):
    """
    Adjust brightness in HSV space to avoid overly dark colors.
    """
    r, g, b = color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = max(v, v_min) * factor
    v = min(v, 1.0)
    return colorsys.hsv_to_rgb(h, s, v)


def generate_image_token(image: Image.Image) -> str:
    """
    Generate a unique token (SHA-256 hash) for a given image.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    return hashlib.sha256(img_bytes.getvalue()).hexdigest()

def draw_text(im, text, pos, scale=0.4, color='auto', font=cv2.FONT_HERSHEY_SIMPLEX, bg_color=(0, 255, 255), blend=0.33, lineType=1):
    """
    Draw text on an image with optional background color and blending.
    """
    text = str(text)
    pos = [int(pos[0]), int(pos[1])]

    if color == 'auto':
        if bg_color is not None:
            color = (0, 0, 0) if ((bg_color[0] + bg_color[1] + bg_color[2]) / 3) > 127.5 else (255, 255, 255)
        else:
            color = (0, 0, 0)

    if bg_color is not None:
        text_size, _ = cv2.getTextSize(text, font, scale, lineType)
        x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
        x_e = int(np.clip(x_s + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
        y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
        y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))

        im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0] * blend + bg_color[0] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1] * blend + bg_color[1] * (1 - blend)
        im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2] * blend + bg_color[2] * (1 - blend)

        pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
        pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))

    cv2.putText(im, text, tuple(pos), font, scale, color, lineType)


def draw_bbox_2d(image, points_2d, color=(0, 255, 0), thickness=2):
    
    points_2d = points_2d.astype(int)
    points_2d = [tuple(point) for point in points_2d]
    # import ipdb;ipdb.set_trace()
    for i in range(4):
        cv2.line(image, points_2d[i], points_2d[(i+1)%4], color, thickness)
    
    for i in range(4, 8):
        cv2.line(image, points_2d[i], points_2d[(i+1)%4+4], color, thickness)
    
    for i in range(4):
        cv2.line(image, points_2d[i], points_2d[i+4], color, thickness)

def predict(image_path, text_prompt):
    with torch.no_grad():
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 生成图像token
        image_token = generate_image_token(img)
        
        # 颜色采样（与deploy.py保持一致）
        pixels = np.array(img).reshape(-1, 3) / 255.0
        brightness = pixels.mean(axis=1)
        prob = brightness / brightness.sum()
        sampled_indices = np.random.choice(pixels.shape[0], 100, p=prob, replace=False)
        sampled_colors = pixels[sampled_indices]
        sampled_colors = sorted(sampled_colors, key=lambda c: colorsys.rgb_to_hsv(*c)[2])
        adjusted_colors = [adjust_brightness(c, factor=2.0, v_min=0.4) for c in sampled_colors]
        
        # GroundingDINO 推理
        image_source_dino, image_dino = convert_image(img)
        boxes, logits, phrases = dino_predict(
            model=dino_model,
            image=image_dino,
            caption=text_prompt,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
        )
        
        if len(boxes) == 0:
            print("No objects found in the image.")
            return img
            
        # 重要修复：使用 image_source_dino 的尺寸
        h, w, _ = image_source_dino.shape  # 修正点
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        
        bbox_2d_list = []
        label_list = []
        for i, box in enumerate(xyxy):
            bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
            label_list.append(phrases[i])
        
        dino_vis_img = img.copy()
        for i, box in enumerate(bbox_2d_list):
            color = (0, 255, 0)
            cv2.rectangle(dino_vis_img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(dino_vis_img, label_list[i], (box[0], max(0, box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        plt.figure(figsize=(12, 8))
        plt.imshow(dino_vis_img)
        plt.axis('off')
        plt.title('Grounding DINO 2D Detection Results')
        plt.tight_layout()
        plt.show()

        # 图像预处理（与deploy.py保持一致）
        raw_img = img.copy()
        original_size = tuple(img.shape[:-1])  # (H, W)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0)
        
        # 应用变换
        img_tensor = sam_trans.apply_image_torch(img_tensor)
        img_tensor = crop_hw(img_tensor)
        before_pad_size = tuple(img_tensor.shape[2:])
        
        img_for_sam = preprocess(img_tensor, cfg).to('cuda:0')
        img_for_dino = preprocess_dino(img_tensor).to('cuda:0')
        
        image_h, image_w = int(before_pad_size[0]), int(before_pad_size[1])
        
        if cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // cfg.model.image_encoder.patch_size, 
                           before_pad_size[1] // cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (cfg.model.pad // cfg.model.image_encoder.patch_size, 
                           cfg.model.pad // cfg.model.image_encoder.patch_size)
        
        # 准备边界框（使用原始图像尺寸进行变换）
        bbox_2d_tensor = torch.tensor(bbox_2d_list)
        bbox_2d_tensor = sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to('cuda:0')
        
        input_dict = {
            "images": img_for_sam,
            'vit_pad_size': torch.tensor(vit_pad_size).to('cuda:0').unsqueeze(0),
            "images_shape": torch.Tensor(before_pad_size).to('cuda:0').unsqueeze(0),
            "image_for_dino": img_for_dino,
            "boxes_coords": bbox_2d_tensor,
        }
        
        # 模型预测
        ret_dict = my_sam_model(input_dict)
        
        K = ret_dict['pred_K']
        decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, cfg, K)
        rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])
        pred_box_ious = ret_dict.get('pred_box_ious', None)

        # print(K)
        # print(rot_mat)
        # print(pred_box_ious)
        
        # print(decoded_bboxes_pred_3d)
        # 可视化（与deploy.py保持一致）
        origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * img_for_sam[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        todo = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
        K = K.detach().cpu().numpy()
        
        # 绘制3D边界框
        for i in range(len(decoded_bboxes_pred_2d)):
            x, y, z, w, h, l, yaw = decoded_bboxes_pred_3d[i].detach().cpu().numpy()
            rot_mat_i = rot_mat[i].detach().cpu().numpy()
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
            vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
            fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))

            color = adjusted_colors[i % len(adjusted_colors)]
            color = [min(255, c*255) for c in color]

            if pred_box_ious is not None:
                best_j = torch.argmax(pred_box_ious[i])
                iou_score = pred_box_ious[i][best_j].item()

            # 绘制3D边界框
            draw_bbox_2d(todo, vertices_2d, color=(int(color[0]), int(color[1]), int(color[2])), thickness=3)

            if label_list[i] is not None:
                dimensions = [round(c, 2) for c in decoded_bboxes_pred_3d[i][3:6].detach().cpu().numpy().tolist()]
                draw_text(todo, f"{label_list[i]} {dimensions}", box_cxcywh_to_xyxy(decoded_bboxes_pred_2d[i]).detach().cpu().numpy().tolist(), scale=0.50*todo.shape[0]/500, bg_color=color)
        
        # 保存和显示
        os.makedirs('./exps/deploy', exist_ok=True)
        cv2.imwrite(f'./exps/deploy/{image_token}.jpg', todo)
        
        # 图像格式转换
        todo = todo / 255
        todo = np.clip(todo, 0.0, 1.0)  # 修复范围
        rgb_image = cv2.cvtColor(todo, cv2.COLOR_BGR2RGB)
        
        # 显示结果
        plt.figure(figsize=(15, 10))
        plt.imshow(rgb_image)
        plt.axis('off')
        plt.title(f'3D Object Detection Results for: {text_prompt}')
        plt.tight_layout()
        plt.show()
        

        results = []
        for i in range(len(decoded_bboxes_pred_3d)):
            results.append({
                "label": label_list[i],
                "bbox_3d": {
                    "x": float(decoded_bboxes_pred_3d[i][0]),
                    "y": float(decoded_bboxes_pred_3d[i][1]),
                    "z": float(decoded_bboxes_pred_3d[i][2]),
                    "width": float(decoded_bboxes_pred_3d[i][3]),
                    "height": float(decoded_bboxes_pred_3d[i][4]),
                    "length": float(decoded_bboxes_pred_3d[i][5]),
                    "yaw": float(decoded_bboxes_pred_3d[i][6])
                }
            })


        return {
            "objects": results,  # list[dict]
            "image": (rgb_image * 255).astype(np.uint8)  # 原可视化图
        }
