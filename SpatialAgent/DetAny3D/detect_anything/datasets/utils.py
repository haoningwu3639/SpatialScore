import torch.nn.functional as F

import cv2
import pickle
import numpy as np
from detect_anything.utils.transforms import ResizeLongestSide
from detect_anything.utils.amg import batched_mask_to_box
from detect_anything.mylogger import *
from copy import deepcopy
from einops import rearrange
import torch
import math
from detect_anything.datasets import transforms_shir as transforms
import hashlib
import matplotlib
from shapely.geometry import Polygon

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def mat2euler(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    #singular = sy < 1e-6

    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], sy)
    z = math.atan2(R[1, 0], R[0, 0])

    return np.array([x, y, z])

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img
def get_depth_transform(cfg=None):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ColorJitter(brightness=0.4,
        #                        contrast=0.4,
        #                        saturation=0.4,
        #                        hue=0.1),
        # transforms.ShortEdgeCenterCrop(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCentralCrop(min_ratio=0.5),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.ShortEdgeCenterCrop(),
        # transforms.RandomCentralCrop(min_ratio=0.5),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test

def coords_gridN(batch, ht, wd, device):
    coords = torch.meshgrid(
        (
            torch.linspace(-1 + 1 / ht, 1 - 1 / ht, ht, device=device),
            torch.linspace(-1 + 1 / wd, 1 - 1 / wd, wd, device=device),
        )
    )

    coords = torch.stack((coords[1], coords[0]), dim=0)[
        None
    ].repeat(batch, 1, 1, 1)
    return coords


def intrinsic2incidence(K, b, h, w, device):
    coords = coords_gridN(b, h, w, device)

    x, y = torch.split(coords, 1, dim=1)
    x = (x + 1) / 2.0 * w
    y = (y + 1) / 2.0 * h

    pts3d = torch.cat([x, y, torch.ones_like(x)], dim=1)
    pts3d = rearrange(pts3d, 'b d h w -> b h w d')
    pts3d = pts3d.unsqueeze(dim=4).to(K.dtype)

    K_ex = K.view([b, 1, 1, 3, 3])
    pts3d = torch.linalg.inv(K_ex) @ pts3d
    pts3d = torch.nn.functional.normalize(pts3d, dim=3)
    return pts3d
    
def compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rotation_matrix=None):
    '''
    output_corners_order: 
        (3) +---------+. (2)
            | ` .     |  ` .
            | (7) +---+-----+ (6)
            |     |   |     |
        (4) +-----+---+. (1)|
            ` .   |     ` . |
            (8) ` +---------+ (5)

    '''
    corners = np.array([
        [ w/2,  h/2,  l/2],
        [ w/2, -h/2,  l/2],
        [-w/2, -h/2,  l/2],
        [-w/2,  h/2,  l/2],
        [ w/2,  h/2, -l/2],
        [ w/2, -h/2, -l/2],
        [-w/2, -h/2, -l/2],
        [-w/2,  h/2, -l/2]
    ])

    fore_plane_center = np.array([
        [0, 0, l/2]
    ])
    if rotation_matrix is not None:
        R = rotation_matrix
    else:
        R = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
    corners_rotated = corners @ R.T
    
    corners_translated = corners_rotated + np.array([x, y, z])

    fore_plane_center = fore_plane_center @ R.T + np.array([x, y, z])

    return corners_translated, fore_plane_center

def project_to_image(points_3d, K):

    points_3d_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    # import ipdb;ipdb.set_trace()
    K_extended = np.hstack((K, np.zeros((3, 1))))
    points_2d = K_extended @ points_3d_homo.T

    new_points_2d = points_2d[:2, :] / points_2d[2:3, :]
    new_points_2d[0, :] = np.sign(points_2d[0, :]) * np.abs(points_2d[0, :]) / np.abs(points_2d[2, :])
    new_points_2d[1, :] = np.sign(points_2d[1, :]) * np.abs(points_2d[1, :]) / np.abs(points_2d[2, :])
    return new_points_2d.T

def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (torch.Tensor): 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img (torch.Tensor): Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].

    Returns:
        torch.Tensor: points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D

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

def compute_3d_bbox_vertices_batch(bboxes, rotation_matrices=None, device='cuda'):
    '''
    Compute 3D bounding box vertices for a batch of boxes using PyTorch.

    Args:
        bboxes (torch.Tensor): Tensor of shape (batch_size, 7), each box represented by [x, y, z, w, h, l, yaw].
        rotation_matrices (torch.Tensor, optional): Tensor of shape (batch_size, 3, 3) with rotation matrices. 
                                                    If None, rotation will be calculated using yaw.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        corners_translated (torch.Tensor): Translated 3D bounding box corners of shape (batch_size, 8, 3).
        fore_plane_centers (torch.Tensor): Center of the front plane of the box, shape (batch_size, 1, 3).
    '''
    # Move inputs to device
    bboxes = bboxes.to(device)
    if rotation_matrices is not None:
        rotation_matrices = rotation_matrices.to(device)

    # Unpack bbox parameters
    x, y, z, w, h, l, yaw = bboxes.split(1, dim=1)  # (batch_size, 1) for each

    # Define corners in local coordinates
    corners = torch.tensor([
        [0.5,  0.5,  0.5],
        [0.5, -0.5,  0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5,  0.5],
        [0.5,  0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5]
    ], device=device)  # (8, 3)

    # Scale corners by w, h, l for each box
    corners = corners.unsqueeze(0) * torch.cat([w, h, l], dim=1).unsqueeze(1)  # (batch_size, 8, 3)

    # Compute rotation matrices from yaw if not provided
    if rotation_matrices is None:
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rotation_matrices = torch.stack([
            torch.stack([cos_yaw, torch.zeros_like(yaw), sin_yaw], dim=-1),
            torch.stack([torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw)], dim=-1),
            torch.stack([-sin_yaw, torch.zeros_like(yaw), cos_yaw], dim=-1)
        ], dim=1).reshape(-1, 3, 3)  # (batch_size, 3, 3)
    # import ipdb;ipdb.set_trace()
    # Rotate and translate corners
    corners_rotated = torch.einsum('bij,bkj->bki', rotation_matrices, corners)  # (batch_size, 8, 3)
    corners_translated = corners_rotated + torch.cat([x, y, z], dim=1).unsqueeze(1)  # (batch_size, 8, 3)

    return corners_translated

def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, eps=eps)

def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    return phi_cos
    
def R_from_allocentric(K, R_view, u=None, v=None):
    """
    Convert a rotation matrix or series of rotation matrices to egocentric
    representation given a 2D location (u, v) in pixels. 
    When u or v are not available, we fall back on the principal point of K.
    """
    if type(K) == torch.Tensor:
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        sx = K[:, 0, 2]
        sy = K[:, 1, 2]

        n = len(K)
        
        oray = torch.stack(((u - sx)/fx, (v - sy)/fy, torch.ones_like(u))).T
        oray = oray / torch.linalg.norm(oray, dim=1).unsqueeze(1)
        angle = torch.acos(oray[:, -1])

        axis = torch.zeros_like(oray)
        axis[:, 0] = axis[:, 0] - oray[:, 1]
        axis[:, 1] = axis[:, 1] + oray[:, 0]
        norms = torch.linalg.norm(axis, dim=1)

        valid_angle = angle > 0

        M = axis_angle_to_matrix(angle.unsqueeze(1)*axis/norms.unsqueeze(1))
        
        R = R_view.clone()
        R[valid_angle] = torch.bmm(M[valid_angle], R_view[valid_angle])

    else:
        fx = K[0][0]
        fy = K[1][1]
        sx = K[0][2]
        sy = K[1][2]
        
        if u is None:
            u = sx

        if v is None:
            v = sy

        oray = np.array([(u - sx)/fx, (v - sy)/fy, 1])
        oray = oray / np.linalg.norm(oray)
        cray = np.array([0, 0, 1])
        angle = math.acos(cray.dot(oray))
        if angle != 0:
            #axis = np.cross(cray, oray)
            axis = np.array([-oray[1], oray[0], 0])
            axis_torch = torch.from_numpy(angle*axis/np.linalg.norm(axis)).float()
            R = np.dot(axis_angle_to_matrix(axis_torch).numpy(), R_view)
        else: 
            R = R_view

    return R

def R_to_allocentric(K, R, u=None, v=None):
    """
    Convert a rotation matrix or series of rotation matrices to allocentric
    representation given a 2D location (u, v) in pixels. 
    When u or v are not available, we fall back on the principal point of K.
    """
    if type(K) == torch.Tensor:
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        sx = K[:, 0, 2]
        sy = K[:, 1, 2]

        n = len(K)
        
        oray = torch.stack(((u - sx)/fx, (v - sy)/fy, torch.ones_like(u))).T
        oray = oray / torch.linalg.norm(oray, dim=1).unsqueeze(1)
        angle = torch.acos(oray[:, -1])

        axis = torch.zeros_like(oray)
        axis[:, 0] = axis[:, 0] - oray[:, 1]
        axis[:, 1] = axis[:, 1] + oray[:, 0]
        norms = torch.linalg.norm(axis, dim=1)

        valid_angle = angle > 0

        M = axis_angle_to_matrix(angle.unsqueeze(1)*axis/norms.unsqueeze(1))
        
        R_view = R.clone()
        R_view[valid_angle] = torch.bmm(M[valid_angle].transpose(2, 1), R[valid_angle])

    else:
        fx = K[0][0]
        fy = K[1][1]
        sx = K[0][2]
        sy = K[1][2]
        
        if u is None:
            u = sx

        if v is None:
            v = sy

        oray = np.array([(u - sx)/fx, (v - sy)/fy, 1])
        oray = oray / np.linalg.norm(oray)
        cray = np.array([0, 0, 1])
        angle = math.acos(cray.dot(oray))
        if angle != 0:
            axis = np.cross(cray, oray)
            axis_torch = torch.from_numpy(angle*axis/np.linalg.norm(axis)).float()
            R_view = np.dot(axis_angle_to_matrix(axis_torch).numpy().T, R)
        else: 
            R_view = R

    return R_view

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def bbox_overlaps_giou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1+area2-inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious,min=-1.0,max = 1.0)
    if exchange:
        ious = ious.T
    return ious

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)    

def generate_instance_id(obj, img_path):
    # 通过拼接一些对象特征生成字符串并计算哈希值
    unique_string = f"{img_path}_{obj['center_cam']}"
    # 生成哈希值，并使用十六进制表示
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def angle2class(angle):
    ''' Convert continuous angle to discrete class and residual. '''
    angle = angle % (2 * torch.pi)
    angle_per_class = 2 * torch.pi / float(12)
    shifted_angle = (angle + angle_per_class / 2) % (2 * torch.pi)
    class_id = torch.floor(shifted_angle / angle_per_class)
    residual = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id.to(torch.long), residual

def class2angle(cls, residual):
    ''' Inverse function to angle2class. '''
    angle_per_class = 2 * np.pi / float(12)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    angle[angle > torch.pi] = angle[angle > torch.pi] - 2 * torch.pi
    return angle

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    poly1 = Polygon(p1).convex_hull      # Polygon：多边形对象
    poly2 = Polygon(p2).convex_hull

    if not poly1.intersects(poly2):
        inter_area = 0  # 如果两四边形不相交
    else:
        inter_area = poly1.intersection(poly2).area  # 相交面积
    return inter_area

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

        (3) +---------+. (2)
            | ` .     |  ` .
            | (7) +---+-----+ (6)
            |     |   |     |
        (4) +-----+---+. (1)|
            ` .   |     ` . |
            (8) ` +---------+ (5)
    '''
    # corner points are in counter clockwise order
    # import ipdb;ipdb.set_trace()
    
    rect1 = np.array([(corners1[i,0], corners1[i,2]) for i in [0,3,4,7]])
    rect2 = np.array([(corners2[i,0], corners2[i,2]) for i in [0,3,4,7]])

    inter_area = convex_hull_intersection(rect1, rect2)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[1,1], corners2[1,1])

    inter_vol = inter_area * max(max(0.0, ymax-ymin), ymin-ymax)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)

    return iou

def undistort_image(image, cam_name, config):
    if cam_name in ['front_left', 'front_center', \
                    'front_right', 'side_left', \
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
                    np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
                    np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
                    np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']
        
        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist,\
                                        D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist, \
                        distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image