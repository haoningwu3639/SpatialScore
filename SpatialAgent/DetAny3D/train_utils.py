import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
import types

from detect_anything.datasets.utils import *
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import open3d as o3d
import math
import argparse

import colorsys

import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def adjust_brightness(color, factor=1.5, v_min=0.3):
    """在 HSV 空间调整亮度，避免过暗"""
    r, g, b = color
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    v = max(v, v_min) * factor  # 强制最低亮度 + 增强
    v = min(v, 1.0)  # 防止过曝
    return colorsys.hsv_to_rgb(h, s, v)

def save_point_cloud(depth_map, images, intrinsic_for_vis, filename, h, w):
    """
    保存点云文件。
    
    参数:
    - depth_map: 深度图数据。
    - origin_img: 用于得到点云上色的彩色图像。
    - intrinsic_for_vis: 相机内参矩阵。
    - filename: 保存点云的文件名。
    - h, w: 图像的高度和宽度。
    """
    # 计算点云的三维坐标
    origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * images[0].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    color_image = origin_img.clone()[:, :h, :w].permute(1,2,0)
    color_image = Image.fromarray(color_image.cpu().numpy().astype('uint8'), 'RGB')

    pred = depth_map.detach().cpu().numpy()
    FX = intrinsic_for_vis[0, 0, 0].cpu().numpy()
    FY = intrinsic_for_vis[0, 1, 1].cpu().numpy()
    BX = intrinsic_for_vis[0, 0, 2].cpu().numpy()
    BY = intrinsic_for_vis[0, 1, 2].cpu().numpy()
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = (x - BX) / FX
    y = (y - BY) / FY
    z = np.array(pred)
    
    # 生成点云的点坐标
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    
    # 获取彩色图像作为点云的颜色
    colors = np.array(color_image).reshape(-1, 3) / 255.0
    
    # 创建并保存点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
def save_color_image(images, h, w, filename):
    """
    保存原始的彩色图像。
    
    参数:
    - data: 原始的图像数据。
    - h, w: 图像的高度和宽度。
    - filename: 保存彩色图像的文件名。
    """
    # 恢复经过归一化处理的彩色图像
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    origin_img = pixel_std * images[0].squeeze(0).detach().cpu() + pixel_mean
    
    # 裁剪到指定大小并转换为图像格式
    color_image = origin_img[:, :h, :w].permute(1, 2, 0)
    color_image = Image.fromarray(color_image.cpu().numpy().astype('uint8'), 'RGB')
    
    # 保存彩色图像
    color_image.save(filename)
def save_depth_image(depth_map, filename, max_depth=None):
    """
    保存深度图为彩色图像。
    
    参数:
    - depth_map: 深度图数据。
    - filename: 保存深度图的文件名。
    - max_depth: 可选，深度图的最大深度值，用于归一化显示。
    """
    # 克隆深度图并对其进行归一化处理
    d = depth_map.clone()

    if max_depth is None:
        max_depth = int(d.max())  # 如果未指定最大深度，则使用深度图中的最大值
    d = colorize(d[0], 0, max_depth)
    # 保存彩色深度图
    Image.fromarray(d).save(filename)

def collector(data):
    ret_dict = {}
    for k in data[0].keys():
        if k == 'prepare_for_dsam' or k == 'img_path':
            ret_dict[k] = data[0][k]
        else:
            k_list = list()
            for unit in data:
                k_list.append(unit[k])
            k_for_batch = torch.stack(k_list)
            ret_dict[k] = k_for_batch
    return ret_dict


def configure_opt(cfg, model, train_loader):

    param_list = list()
    param_list_unidepth = list()

    param_list += list(model.module.sam.mask_decoder.parameters())
    if cfg.tune_with_depth:
        param_list_unidepth += list(model.module.sam.image_encoder.depth_head.parameters()) 
        param_list_unidepth += list(
            model.module.sam.image_encoder.dino.parameters()
        )
    param_list +=  list(model.module.sam.image_encoder.spm.parameters()) + list(model.module.sam.image_encoder.interactions.parameters()) \
        + list(model.module.sam.image_encoder.up.parameters())
    
    optimizer = torch.optim.AdamW(param_list, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)

    optimizer.add_param_group({'params': model.module.sam.image_encoder.cls_token})
    optimizer.add_param_group({'params': model.module.sam.image_encoder.level_embed})
    
    optimizer.add_param_group({
        'params': param_list_unidepth,
        'lr': cfg.opt.unidepth_lr,  # 自定义学习率
        'weight_decay': cfg.opt.unidepth_weight_decay,  # 自定义权重衰减
    })

    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max =  cfg.num_epochs)
    
    return optimizer, scheduler

def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)

def SILogLoss(input, target, coefficient, masks = None, log_mode = False):

    if masks is not None:
        input = input[masks > 0]
        target = target[masks > 0]
    if input.numel() == 0:
        return None
    if log_mode:
        alpha = 1e-7
        g = torch.log(input + alpha) - torch.log(target + alpha)
    else:
        g = input - target

    Dg = torch.pow(g, 2).mean() - coefficient * torch.pow(torch.mean(g), 2)
    loss = torch.sqrt(Dg)

    if torch.isnan(loss):
        print("Nan SILog loss")
        print("Input min max", torch.min(input), torch.max(input))
    return loss

def save_mask_images(pred_masks, iou_predictions, image_h, image_w, gt_bboxes_2d, images, instance_ids, save_root, file_name = None, debug_mode = False):
    """Save the predicted masks as images."""
    origin_img = torch.Tensor(
            [58.395, 57.12, 57.375]).view(-1, 1, 1) * images[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    origin_img = cv2.cvtColor(origin_img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
    pred_masks = pred_masks[..., :image_h, :image_w]
    
    for i in range(pred_masks.shape[0]):
        if file_name == None:
            to_save_file_name = f'{instance_ids[i]}'
        if debug_mode:  
            [bbox_x1, bbox_y1, bbox_x2, bbox_y2] = gt_bboxes_2d[i]
            coor = [(int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2))]
            for j in range(pred_masks[i].shape[0]):  # Assuming three channels to save
                to_draw_masks = pred_masks[i][j].clone() > 0
                to_draw_masks = to_draw_masks.detach().cpu().numpy()
                to_draw_image = origin_img.copy() * 0.5 + 255 * np.repeat(to_draw_masks[:, :, np.newaxis], 3, axis=-1)
                cv2.rectangle(to_draw_image, coor[0], coor[1], (0, 0, 255), 2)
                cv2.putText(to_draw_image, f'iou_score: {int(iou_predictions[i][j] * 1000) / 1000}', (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
                cv2.imwrite(f'{save_root}/{to_save_file_name}_{i}_{j}.jpg', to_draw_image)
        else:
            to_draw_masks = pred_masks[i][-1].clone() > 0.8
            to_draw_masks = 255 * to_draw_masks.detach().cpu().numpy()
            cv2.imwrite(f'{save_root}/{to_save_file_name}_mask.jpg', to_draw_masks)

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

def vis_prompt_func(cfg, images, point_coords, K, gt_K, image_h, image_w, pred_masks, gt_bboxes, bboxes_pred_2d, gt_bboxes_3d, bboxes_pred_3d, gt_center_2d, pred_center_2d, gt_rot_mat, rot_mat, dataset_name, epoch, iter, pred_box_ious=None):
    
    # Prepare the intrinsic matrix
    K = K.cpu().numpy()
    
    # Prepare the image
    origin_img = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1) * images[0, :, :image_h, :image_w].squeeze(0).detach().cpu() + torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    todo = cv2.cvtColor(origin_img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    
    pixels = np.array(origin_img.permute(1, 2, 0).numpy()).reshape(-1, 3) / 255.0
    pixels = np.clip(pixels, 0, 1)
    # 改进点1：根据亮度加权采样（避免全随机选到过多暗色）
    brightness = pixels.mean(axis=1)  # 计算每个像素的亮度
    prob = brightness / brightness.sum()  # 亮度越高采样概率越大
    sampled_indices = np.random.choice(pixels.shape[0], 100, p=prob, replace=False)
    sampled_colors = pixels[sampled_indices]

    # 改进点2：按亮度排序而非直接排序
    sampled_colors = sorted(sampled_colors, key=lambda c: colorsys.rgb_to_hsv(*c)[2])

    # 应用亮度增强
    adjusted_colors = [adjust_brightness(c, factor=2.0, v_min=0.4) for c in sampled_colors]

    
    for i in range(point_coords.shape[0]):
        # Get 3D bounding box parameters
        x, y, z, w, h, l, yaw = bboxes_pred_3d[i].detach().cpu().numpy()
        if rot_mat is not None:
            rot_mat_i = rot_mat[i].cpu().numpy()
        else:
            rot_mat_i = None
        # Compute the 3D vertices and 2D projections
        vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
        vertices_2d = project_to_image(vertices_3d, K.squeeze(0))
        fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze(0))
        
        # Choose the color for the current bounding box
        color = adjusted_colors[i]
        color = [min(255, c*255) for c in color]
        
        # If IoU scores are provided, get the best IoU for the current object
        if pred_box_ious is not None:
            best_j = torch.argmax(pred_box_ious[i])  # Get the index of the best IoU box
            iou_score = pred_box_ious[i][best_j].item()  # Get the IoU score as a scalar
            
            # Only continue if IoU is above a certain threshold (e.g., 0.3)
            if iou_score < 0.05:
                continue

        # Draw the 2D bounding box (predicted)
        draw_bbox_2d(todo, vertices_2d, color=color)
        cv2.circle(todo, fore_plane_center_2d[0].astype(int), 2, color , 2)
    # Save the image with bounding boxes and IoU annotations
    cv2.imwrite(f'{cfg.exp_dir}/{dataset_name}_vis_pic_{iter}_3D.png', todo)

def chamfer_loss(vals, target):
    B = vals.shape[0]
    xx = vals.view(B, 8, 1, 3)
    yy = target.view(B, 1, 8, 3)
    l1_dist = (xx - yy).abs().sum(-1)
    l1 = (l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1))
    return l1


def calculate_pred_pose_for_cubercnn(pred_bbox_3d, pred_rot_mat, device_id):
    R_inv = torch.tensor(
        [[-0., -0., -1.],
        [ 0.,  1.,  0.],
        [ 1.,  0.,  0.]]).to(device_id).to(pred_bbox_3d.dtype)
    if pred_rot_mat is not None:
        pred_pose_for_cubercnn = torch.matmul(pred_rot_mat, R_inv)
    else:
        yaws = pred_bbox_3d[..., 6:].cpu().numpy()
        yaw_only_rot_mat = np.zeros((yaws.shape[0], 3, 3))

        # 遍历每个 yaw 值，计算旋转矩阵
        for i in range(yaws.shape[0]):
            angle = yaws[i, 0]
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
            yaw_only_rot_mat[i] = rotation_matrix
        yaw_only_rot_mat = torch.tensor(yaw_only_rot_mat).to(device_id).to(pred_bbox_3d.dtype)
        pred_pose_for_cubercnn = torch.matmul(yaw_only_rot_mat, R_inv)
    return pred_pose_for_cubercnn

def decode_bboxes(ret_dict, cfg, K_for_convert):
    """
    Decode 2D and 3D bounding boxes from prediction results.

    Args:
        ret_dict (dict): The prediction dictionary containing keys:
            - 'pred_bbox_2d': Predicted 2D bounding boxes.
            - 'pred_bbox_3d_alpha_cls': Predicted alpha class.
            - 'pred_bbox_3d_alpha_res': Predicted alpha residual.
            - 'pred_center_2d': Predicted 2D center.
            - 'pred_bbox_3d_depth': Predicted 3D depth.
            - 'pred_bbox_3d_dims': Predicted 3D dimensions.
        cfg (dict): Configuration containing model settings.
        K_for_convert (Tensor): Camera intrinsic matrix for converting image points to 3D space.

    Returns:
        decoded_bboxes_pred_2d (Tensor): The decoded 2D bounding boxes.
        decoded_bboxes_pred_3d (Tensor): The decoded 3D bounding boxes.
    """
    # Decode 2D bounding boxes
    bboxes_pred_2d = ret_dict['pred_bbox_2d']
    decoded_bboxes_pred_2d = bboxes_pred_2d * cfg.model.pad

    # Decode 3D rotation (alpha angle)
    pred_alpha_cls = torch.argmax(ret_dict['pred_bbox_3d_alpha_cls'], dim=-1)
    pred_alpha_res = ret_dict['pred_bbox_3d_alpha_res'][np.arange(pred_alpha_cls.shape[0]), pred_alpha_cls]
    
    # Assuming class2angle function is already defined and accessible
    pred_alpha = class2angle(pred_alpha_cls, pred_alpha_res)

    # Compute predicted yaw (rotation around Y axis)
    bboxes_pred_2d_center_x = decoded_bboxes_pred_2d[..., 0]
    bboxes_pred_2d_center_y = decoded_bboxes_pred_2d[..., 1]
    pred_ry = torch.atan2(
        bboxes_pred_2d_center_x - K_for_convert[..., 0, 2], K_for_convert[..., 0, 0]) + pred_alpha
    pred_ry = (pred_ry + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize the angle to [-pi, pi]
    pred_ry = pred_ry.unsqueeze(-1)

    # Decode 3D center with depth
    pred_centers_2d_with_depth = torch.cat([ret_dict['pred_center_2d'] * cfg.model.pad, ret_dict['pred_bbox_3d_depth'].exp()], dim=-1)
    
    # Assuming points_img2cam function is already defined and accessible
    pred_centers_3d = points_img2cam(pred_centers_2d_with_depth, K_for_convert[0])

    # Decode 3D bounding box dimensions
    decoded_bboxes_pred_3d = torch.cat([pred_centers_3d, ret_dict['pred_bbox_3d_dims'].exp(), pred_ry], dim=-1)

    return decoded_bboxes_pred_2d, decoded_bboxes_pred_3d

def decode_bboxes_virtual_to_real(ret_dict, cfg, K_gt, K_pred):
    """
    Decode 2D and 3D bounding boxes from prediction results.

    Args:
        ret_dict (dict): The prediction dictionary containing keys:
            - 'pred_bbox_2d': Predicted 2D bounding boxes.
            - 'pred_bbox_3d_alpha_cls': Predicted alpha class.
            - 'pred_bbox_3d_alpha_res': Predicted alpha residual.
            - 'pred_center_2d': Predicted 2D center.
            - 'pred_bbox_3d_depth': Predicted 3D depth.
            - 'pred_bbox_3d_dims': Predicted 3D dimensions.
        cfg (dict): Configuration containing model settings.
        K_for_convert (Tensor): Camera intrinsic matrix for converting image points to 3D space.

    Returns:
        decoded_bboxes_pred_2d (Tensor): The decoded 2D bounding boxes.
        decoded_bboxes_pred_3d (Tensor): The decoded 3D bounding boxes.
    """
    # Decode 2D bounding boxes
    bboxes_pred_2d = ret_dict['pred_bbox_2d']
    decoded_bboxes_pred_2d = bboxes_pred_2d * cfg.model.pad

    # Decode 3D rotation (alpha angle)
    pred_alpha_cls = torch.argmax(ret_dict['pred_bbox_3d_alpha_cls'], dim=-1)
    pred_alpha_res = ret_dict['pred_bbox_3d_alpha_res'][np.arange(pred_alpha_cls.shape[0]), pred_alpha_cls]
    
    # Assuming class2angle function is already defined and accessible
    pred_alpha = class2angle(pred_alpha_cls, pred_alpha_res)

    # Compute predicted yaw (rotation around Y axis)
    bboxes_pred_2d_center_x = decoded_bboxes_pred_2d[..., 0]
    bboxes_pred_2d_center_y = decoded_bboxes_pred_2d[..., 1]
    pred_ry = torch.atan2(
        bboxes_pred_2d_center_x - K_pred[..., 0, 2], K_pred[..., 0, 0]) + pred_alpha
    pred_ry = (pred_ry + torch.pi) % (2 * torch.pi) - torch.pi  # Normalize the angle to [-pi, pi]
    pred_ry = pred_ry.unsqueeze(-1)
    K_gt = K_gt.to(ret_dict['pred_bbox_3d_depth'].device)
    # import ipdb;ipdb.set_trace()
    real_depth = torch.sqrt((K_gt[..., 1, 1] * K_gt[..., 0, 0]) / (K_pred[..., 0, 0] * K_pred[..., 0, 0])) * ret_dict['pred_bbox_3d_depth'].exp()
    # Assuming points_img2cam function is already defined and accessible
    # real_depth_x = K_gt[..., 0, 0] / K_pred[..., 0, 0] * ret_dict['pred_bbox_3d_depth'].exp()
    # real_depth_y = K_gt[..., 1, 1] / K_pred[..., 1, 1] * ret_dict['pred_bbox_3d_depth'].exp()
    # # Decode 3D center with depth
    # pred_centers_2d_with_depth_x = torch.cat([ret_dict['pred_center_2d'] * cfg.model.pad, real_depth_x], dim=-1)
    # pred_centers_2d_with_depth_y = torch.cat([ret_dict['pred_center_2d'] * cfg.model.pad, real_depth_y], dim=-1)
    # pred_centers_2d_with_depth = torch.cat([pred_centers_2d_with_depth_x[..., :1], pred_centers_2d_with_depth_y[..., 1:2], real_depth], dim=-1)

    # Decode 3D center with depth
    pred_centers_2d_with_depth = torch.cat([ret_dict['pred_center_2d'] * cfg.model.pad, real_depth], dim=-1)
    # new_K = torch.tensor([[int(K_gt[0, 0, 0]), 0, K_pred[0, 0, 2].item()],
    #                   [0, int(K_gt[0, 1, 1]), K_pred[0, 1, 2].item()],
    #                   [0, 0, 1]]).to(ret_dict['pred_bbox_3d_depth'].device)
    pred_centers_3d = points_img2cam(pred_centers_2d_with_depth, K_gt[0])

    # Decode 3D bounding box dimensions
    decoded_bboxes_pred_3d = torch.cat([pred_centers_3d, ret_dict['pred_bbox_3d_dims'].exp(), pred_ry], dim=-1)

    return decoded_bboxes_pred_2d, decoded_bboxes_pred_3d

