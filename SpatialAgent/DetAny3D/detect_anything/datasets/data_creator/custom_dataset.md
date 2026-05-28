## 🧪 How to Develop a Customized Dataset for DetAny3D

This section describes how to prepare a customized dataset for use with DetAny3D. The data preparation pipeline converts your dataset into a standard `.pkl` format, which can be directly used for inference or training.

### 🧱 1. Expected Output Format (`.pkl`)

Each `.pkl` file should be a list of dictionary samples with the following keys:

```python
{
  "img_path": str,              # Path to the RGB image
  "depth_path": Optional[str],  # Path to the depth map (can be None)
  "K": np.ndarray,              # Camera intrinsic matrix, shape (1, 3, 3)
  "obj_list": [                 # List of 3D object annotations
    {
      "3d_bbox": [x, y, z, w, h, l, yaw],     # 3D bounding box in camera coordinates
      "2d_bbox_proj": [x1, y1, x2, y2],       # (optional) Projected 2D bounding box
      "2d_bbox_tight": [x1, y1, x2, y2],                # (optional) Tight 2D box
      "2d_bbox_trunc": [x1, y1, x2, y2],                # (optional) Truncated 2D box
      "label": int,                          # Category ID
      "rotation_pose": np.ndarray,           # 3×3 rotation matrix 
      "instance_id": str,                    # Unique ID per object
      "score": float,                        # Confidence score (1.0 for GT)
      "image_id": int,                       # Unique image ID
      "visibility": float,                   # (optional) GT visibility
      "truncation": float,                   # (optional) GT truncation
    }
  ]
}
```

---

### 🖼️ 2. Coordinate System & Yaw Definition

```
           up   z (forward)   yaw = 0
           |     ↗
           |    /
           |   /
           |  /
           | /
           |/
left  ---- O ----→ x (right)   yaw = π/2
           |
           |
           |
           |
           ↓
         y (down)
```

- Coordinate system: OpenCV-style camera frame
  - +X → right
  - +Y → down
  - +Z → forward (view direction)
- **Yaw** is rotation around the **Y axis (downward)**
  - **yaw = 0** → object faces forward (+Z)
  - **yaw = π/2** → object turns right, faces +X


---

### 🌀 3. Rotation Pose Construction

If your dataset only provides a **yaw angle** (as is common in autonomous driving datasets), you can construct the full 3×3 `rotation_pose` matrix using the following logic:

```python
import numpy as np

pose = np.array([
    [np.cos(yaw + np.pi), 0, np.sin(yaw + np.pi)],
    [0, 1, 0],
    [-np.sin(yaw + np.pi), 0, np.cos(yaw + np.pi)]
])

# Flip Z axis to match camera coordinate convention
diag_matrix = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]
])

rotation_pose = np.dot(diag_matrix, pose.T)
```

#### TODO:
If your dataset provides a full **3-DoF rotation matrix**, we will update this guide soon to support direct usage.

---

### ✅ 4. Pose Validation (Optional)

You can visually check your 3D box and pose using the following snippet, based on `detect_anything.datasets.utils.compute_3d_bbox_vertices`:

```python
# Visualization example
to_draw = cv2.imread(img_path)

# Project 2D box
[bbox_x1, bbox_y1, bbox_x2, bbox_y2] = 2d_bbox_proj
cv2.rectangle(to_draw, (int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2)), (0, 0, 255), 2)

# Project 3D bbox
x, y, z, w, h, l, yaw = 3d_bbox
vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rotation_pose)  # rotation_pose=None if only yaw

vertices_2d = project_to_image(vertices_3d, K.squeeze())
fore_plane_center_2d = project_to_image(fore_plane_center_3d, K.squeeze())

draw_bbox_2d(to_draw, vertices_2d)
cv2.circle(to_draw, fore_plane_center_2d[0].astype(int), 2, (0, 0, 255), 1)
cv2.imwrite('img_with_3d_gt.png', to_draw)
```

> This helps ensure that your pose and 3D geometry are correctly aligned with the projection model.

---

### 🌊 5. Depth Format & Usage

DetAny3D supports training and inference **with or without depth**.

#### ✅ No Depth Available?
If your dataset does not provide depth maps, simply set:
```python
depth_path = None
```
The system will automatically use a zero-depth placeholder. 

---

#### ✅ If Depth is Available:

You can provide depth maps in either `.npy`, `.png`, or `.hdf5` format:

| Format      | Shape     | Unit     | Notes                                 |
|-------------|-----------|----------|---------------------------------------|
| `.npy`      | `(H, W)`  | meters   | Metric depth in float32               |
| `.png`      | `(H, W)`  | uint16   | Needs to be rescaled via `metric_scale` in config |
| `.hdf5`     | varies    | meters   | Only supported for `hypersim` for now        |

- For `.npy` format: set `metric_scale = 1` in your config
- For `.png` format: set the appropriate `metric_scale` (e.g., 80.0) to decode back to meters



