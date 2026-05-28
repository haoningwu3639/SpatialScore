import re

SPATIALCORPUS_SPATIAL_MAP = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_spatial_map.json",
    "data_path": "",
}

SPATIALCORPUS_2D_ROTATION = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_2d_rotation.json",
    "data_path": "",
}

SPATIALCORPUS_3D_ROTATION = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_3d_rotation.json",
    "data_path": "",
}

SPATIALCORPUS_VIEW_MATCHING = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_view_matching.json",
    "data_path": "",
}

SPATIALCORPUS_VIEW_IDENTIFICATION = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_view_identification.json",
    "data_path": "",
}

SPATIALCORPUS_ABSOLUTE_DEPTH = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_absolute_depth.json",
    "data_path": "",
}

SPATIALCORPUS_ABSOLUTE_DISTANCE = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_absolute_distance.json",
    "data_path": "",
}

SPATIALCORPUS_CAMERA_EXTRINSICS = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_camera_extrinsics.json",
    "data_path": "",
}

SPATIALCORPUS_CAMERA_INTRINSICS = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_camera_intrinsics.json",
    "data_path": "",
}

SPATIALCORPUS_CAMERA_MOTION = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_camera_motion.json",
    "data_path": "",
}

SPATIALCORPUS_HOMOGRAPHY = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_homography.json",
    "data_path": "",
}

SPATIALCORPUS_OBJECT_EXISTENCE = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_object_existence.json",
    "data_path": "",
}

SPATIALCORPUS_ABSOLUTE_SIZE = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_absolute_size.json",
    "data_path": "",
}

SPATIALCORPUS_POINT_TRACKING = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_point_tracking.json",
    "data_path": "",
}

SPATIALCORPUS_RECONSTRUCTION = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_reconstruction.json",
    "data_path": "",
}

SPATIALCORPUS_RELATIVE_DEPTH = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_relative_depth.json",
    "data_path": "",
}

SPATIALCORPUS_RELATIVE_DISTANCE = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_relative_distance.json",
    "data_path": "",
}


SPATIALCORPUS_RELATIVE_SIZE = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_relative_size.json",
    "data_path": "",
}

SPATIALCORPUS_THREED_OBJECT_DETECTION = {
    "annotation_path": "./SpatialCorpus/jsons/SpatialCorpus_threed_object_detection.json",
    "data_path": "",
}

data_dict = {
    "SpatialCorpus_2D_Rotation": SPATIALCORPUS_2D_ROTATION,
    "SpatialCorpus_3D_Rotation": SPATIALCORPUS_3D_ROTATION,
    "SpatialCorpus_Spatial_Map": SPATIALCORPUS_SPATIAL_MAP,
    "SpatialCorpus_Absolute_Depth": SPATIALCORPUS_ABSOLUTE_DEPTH,
    "SpatialCorpus_Absolute_Distance": SPATIALCORPUS_ABSOLUTE_DISTANCE,
    "SpatialCorpus_Camera_Extrinsics": SPATIALCORPUS_CAMERA_EXTRINSICS,
    "SpatialCorpus_Camera_Intrinsics": SPATIALCORPUS_CAMERA_INTRINSICS,
    "SpatialCorpus_Camera_Motion": SPATIALCORPUS_CAMERA_MOTION,
    "SpatialCorpus_Homography": SPATIALCORPUS_HOMOGRAPHY,
    "SpatialCorpus_3D_Object_Detection": SPATIALCORPUS_THREED_OBJECT_DETECTION,
    "SpatialCorpus_Point_Tracking": SPATIALCORPUS_POINT_TRACKING,
    "SpatialCorpus_Relative_Depth": SPATIALCORPUS_RELATIVE_DEPTH,
    "SpatialCorpus_Relative_Size": SPATIALCORPUS_RELATIVE_SIZE,    
    
    "SpatialCorpus_View_Matching": SPATIALCORPUS_VIEW_MATCHING,
    "SpatialCorpus_View_Identification": SPATIALCORPUS_VIEW_IDENTIFICATION,
    "SpatialCorpus_Object_Existence": SPATIALCORPUS_OBJECT_EXISTENCE,
    "SpatialCorpus_Absolute_Size": SPATIALCORPUS_ABSOLUTE_SIZE,
    "SpatialCorpus_Reconstruction": SPATIALCORPUS_RECONSTRUCTION,
    "SpatialCorpus_Relative_Distance": SPATIALCORPUS_RELATIVE_DISTANCE,
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = list(data_dict.keys())
    
    print(f"Testing {len(dataset_names)} datasets...")
    configs = data_list(dataset_names)
    for i, config in enumerate(configs):
        print(f"[{i+1}] {dataset_names[i]}: {config['annotation_path']}")