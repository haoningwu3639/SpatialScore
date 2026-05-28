import os
import sys
import shutil
import tempfile
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import torch
from PIL import Image
from pillow_heif import register_heif_opener

try:
    from mapanything.utils.geometry import depthmap_to_world_frame
    from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_model
    from mapanything.utils.image import load_images

except ImportError as e:
    sys.exit(1)

register_heif_opener()


class MapAnythingEstimator:
    def __init__(self):
        print("Initializing MapAnythingEstimator...")
        print("Loading configuration...")
        self.config = {
            "path": "configs/train.yaml",
            "hf_model_name": "/remote-home/share/huggingface/map-anything",
            "model_str": "mapanything",
            "config_overrides": [
                "machine=aws",
                "model=mapanything",
                "model/task=images_only",
                "model.encoder.uses_torch_hub=false",
            ],
            "checkpoint_name": "model.safetensors",
            "config_name": "config.json",
            "trained_with_amp": True,
            "trained_with_amp_dtype": "bf16",
            "data_norm_type": "dinov2",
            "patch_size": 14,
            "resolution": 518,
        }
        
        script_dir_tool = os.path.dirname(os.path.abspath(__file__)) # .../map_anything/scripts
        project_root_map_anything = os.path.abspath(os.path.join(script_dir_tool, '..')) # .../map_anything
        config_dir_abs = os.path.join(project_root_map_anything, 'configs') # .../map_anything/configs

        if not os.path.isdir(config_dir_abs):
             raise NotADirectoryError(
                 f"Hydra config directory for map-anything not found. "
                 f"Expected at: {config_dir_abs}"
             )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading MapAnything model...")
        original_cwd = os.getcwd()
        try:
            os.chdir(project_root_map_anything)
            print(f"Temporarily changed CWD to: {project_root_map_anything}")

            self.model = initialize_mapanything_model(self.config, self.device)

        except Exception as e:
             os.chdir(original_cwd)
             print(f"Restored CWD to: {original_cwd}")
             raise e
        finally:
             os.chdir(original_cwd)
             print(f"Restored CWD to: {original_cwd}")

        self.model.eval()
        print("Model loaded successfully.")

        self._cache = {}
        self._cache_key = None

    def _run_inference(self, image_path: str):
        
        abs_image_path = os.path.abspath(image_path)
        
        if self._cache_key == abs_image_path and "depth" in self._cache:
            return self._cache["depth"], self._cache["points3d"]

        print(f"Running inference on: {abs_image_path}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            
            target_dir_images = os.path.join(tmpdir, "images")
            os.makedirs(target_dir_images)

            try:
                file_ext = os.path.splitext(image_path)[1].lower()
                if file_ext in [".heic", ".heif"]:
                    with Image.open(image_path) as img:
                        if img.mode not in ("RGB", "L"):
                            img = img.convert("RGB")
                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        dst_path = os.path.join(target_dir_images, f"{base_name}.jpg")
                        img.save(dst_path, "JPEG", quality=95)
                else:
                    dst_path = os.path.join(target_dir_images, os.path.basename(image_path))
                    shutil.copy(image_path, dst_path)
            except Exception as e:
                raise IOError(f"无法读取或转换图像 {image_path}: {e}")
            
            if not os.path.exists(dst_path):
                 raise FileNotFoundError(f"未能成功创建临时图像文件 at {dst_path}")

            views = load_images(target_dir_images) 
            if not views:
                raise ValueError(f"MapAnything's load_images 未能加载: {image_path}")

            with torch.no_grad():
                outputs = self.model.infer(
                    views, 
                    apply_mask=True, 
                    mask_edges=True, 
                    memory_efficient_inference=False
                )
            
            pred = outputs[0] 
            depth_map_torch = pred["depth_z"][0].squeeze(-1) # (H, W)
            intrinsics_torch = pred["intrinsics"][0]         # (3, 3)
            camera_pose_torch = pred["camera_poses"][0]      # (4, 4)
            
            pts3d_computed, valid_mask = depthmap_to_world_frame(
                depth_map_torch, intrinsics_torch, camera_pose_torch
            )

            depth_map_np = depth_map_torch.cpu().numpy()
            pts3d_map_np = pts3d_computed.cpu().numpy() # (H, W, 3)

            self._cache = {
                "depth": depth_map_np, 
                "points3d": pts3d_map_np,
            }
            self._cache_key = abs_image_path

            return depth_map_np, pts3d_map_np

    def get_depth_map(self, image_path: str) -> np.ndarray:
        """
        获取单张图像的公制深度图 (metric-based)。
        """
        depth_map, _ = self._run_inference(image_path)
        return depth_map

    def get_distance_between_points(self, image_path: str, p1_coords: tuple, p2_coords: tuple, original_shape_WH: tuple) -> float:
        """
        计算图像中两个像素点对应的 3D 空间绝对距离。
        
        参数:
            image_path (str): 输入图像的路径。
            p1_coords (tuple): 第一个点的 (x, y) 像素坐标。
            p2_coords (tuple): 第二个点的 (x, y) 像素坐标。
            original_shape_WH (tuple): 原始图像的 (width, height) 元组。
            
        返回:
            float: 两个 3D 点之间的距离，单位为米（meters）。
        """
        _, pts3d_map = self._run_inference(image_path)
        
        original_w, original_h = original_shape_WH
        model_h, model_w, _ = pts3d_map.shape
        
        x1_orig, y1_orig = p1_coords
        x2_orig, y2_orig = p2_coords

        x1_scaled = int((x1_orig / original_w) * model_w)
        y1_scaled = int((y1_orig / original_h) * model_h)
        x2_scaled = int((x2_orig / original_w) * model_w)
        y2_scaled = int((y2_orig / original_h) * model_h)

        x1_scaled = max(0, min(x1_scaled, model_w - 1))
        y1_scaled = max(0, min(y1_scaled, model_h - 1))
        x2_scaled = max(0, min(x2_scaled, model_w - 1))
        y2_scaled = max(0, min(y2_scaled, model_h - 1))

        point1_3d = pts3d_map[y1_scaled, x1_scaled]
        point2_3d = pts3d_map[y2_scaled, x2_scaled]

        distance = np.linalg.norm(point1_3d - point2_3d)
        
        print(f"坐标 {p1_coords} (原图) -> 缩放至 ({x1_scaled}, {y1_scaled}) (模型图)")
        print(f"坐标 {p2_coords} (原图) -> 缩放至 ({x2_scaled}, {y2_scaled}) (模型图)")
        
        return float(distance)
        
    def clear_cache(self):
        """清空缓存"""
        self._cache = {}
        self._cache_key = None

if __name__ == "__main__":
    
    YOUR_IMAGE_PATH = os.path.join(project_root, "examples/005.jpeg")
    
    if not os.path.exists(YOUR_IMAGE_PATH):
        print(f"错误: 示例图片未找到: {YOUR_IMAGE_PATH}")
        print("请编辑 'YOUR_IMAGE_PATH' 变量，指向一个有效的图片文件。")
    else:
        print("--- 正在初始化模型 (若第一次运行，需下载模型)... ---")
        start_init = time.time()
        estimator = MapAnythingEstimator()
        print(f"--- 模型初始化完成，耗时: {time.time() - start_init:.2f} 秒 ---")
        
        
        print(f"\n--- [测试 1] 正在为 {os.path.basename(YOUR_IMAGE_PATH)} 生成深度图... ---")
        start_depth = time.time()
        metric_depth = estimator.get_depth_map(YOUR_IMAGE_PATH)
        print(f"--- 深度图生成完毕，耗时: {time.time() - start_depth:.2f} 秒 ---")
        
        print(f"深度图形状: {metric_depth.shape}")
        print(f"深度值 (米): Min={metric_depth.min():.2f}m, Max={metric_depth.max():.2f}m, Avg={metric_depth.mean():.2f}m")

        try:
            depth_vis = metric_depth.copy()
            v_min, v_max = np.percentile(depth_vis[depth_vis > 0], [5, 95])
            depth_vis = np.clip(depth_vis, v_min, v_max)
            depth_vis = (depth_vis - v_min) / (v_max - v_min)
            
            depth_vis = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            depth_vis[metric_depth == 0] = 0 # 无效深度设为黑色
            
            output_depth_path = "output_depth_map.png"
            cv2.imwrite(output_depth_path, depth_vis)
            print(f"可视化的深度图已保存到: {os.path.abspath(output_depth_path)}")
            
        except Exception as e:
            print(f"保存可视化深度图失败: {e}")

        print(f"\n--- [测试 2] 正在读取 {os.path.basename(YOUR_IMAGE_PATH)} 的原始尺寸... ---")
        try:
            with Image.open(YOUR_IMAGE_PATH) as img:
                original_shape_WH = img.size # (width, height)
            print(f"原始尺寸 (W, H): {original_shape_WH}")
        except Exception as e:
            print(f"无法读取图像尺寸: {e}")
            original_shape_WH = None

        if original_shape_WH:
            point_a_xy = (100, 100) # (x, y)
            point_b_xy = (1000, 1000) # (x, y)

            print(f"--- 正在计算 {point_a_xy} 和 {point_b_xy} 之间的 3D 距离... ---")
            
            start_dist = time.time()
            distance = estimator.get_distance_between_points(
                YOUR_IMAGE_PATH, 
                point_a_xy, 
                point_b_xy,
                original_shape_WH 
            )
            print(f"--- 距离计算完毕，耗时: {time.time() - start_dist:.4f} 秒 (使用了缓存) ---")
            
            print(f"像素点 {point_a_xy} 和 {point_b_xy} 之间的 3D 绝对距离是: {distance:.4f} 米")

            try:
                img = cv2.imread(YOUR_IMAGE_PATH)
                color = (0, 0, 255)
                thickness = 2
                
                cv2.circle(img, point_a_xy, 5, color, -1) 
                cv2.circle(img, point_b_xy, 5, color, -1)
                cv2.line(img, point_a_xy, point_b_xy, color, thickness)
                
                output_points_path = "output_points_visualization.png"
                cv2.imwrite(output_points_path, img)
                print(f"带有点的可视化图像已保存到: {os.path.abspath(output_points_path)}")

            except Exception as e:
                print(f"保存点可视化图像失败: {e}")
