#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task definitions and configurations for Rex Omni
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class TaskType(Enum):
    """Supported task types"""

    DETECTION = "detection"
    POINTING = "pointing"
    VISUAL_PROMPTING = "visual_prompting"
    KEYPOINT = "keypoint"
    OCR_BOX = "ocr_box"
    OCR_POLYGON = "ocr_polygon"
    GUI_DETECTION = "gui_grounding"
    GUI_POINTING = "gui_pointing"


@dataclass
class TaskConfig:
    """Configuration for a specific task"""

    name: str
    prompt_template: str
    description: str
    output_format: str
    requires_categories: bool = True
    requires_visual_prompt: bool = False
    requires_keypoint_type: bool = False


# Task configurations
TASK_CONFIGS: Dict[TaskType, TaskConfig] = {
    TaskType.DETECTION: TaskConfig(
        name="Detection",
        prompt_template="Detect {categories}. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        description="Detect objects and return in bounding box format",
        output_format="boxes",
        requires_categories=True,
    ),
    TaskType.POINTING: TaskConfig(
        name="Pointing",
        prompt_template="Point to {categories}.",
        description="Point to objects and return in point format",
        output_format="points",
        requires_categories=True,
    ),
    TaskType.VISUAL_PROMPTING: TaskConfig(
        name="Visual Prompting",
        prompt_template="Given reference boxes {visual_prompt} indicating one or more objects, find all similar objects in the image and output their bounding boxes.",
        description="Ground visual prompts to image regions",
        output_format="boxes",
        requires_categories=False,
        requires_visual_prompt=True,
    ),
    TaskType.KEYPOINT: TaskConfig(
        name="Keypoint",
        prompt_template="Can you detect each {categories} in the image using a [x0, y0, x1, y1] box format, and then provide the coordinates of its {keypoints} as [x0, y0]? Output the answer in JSON format.",
        description="Detect keypoints for specific object types",
        output_format="keypoints",
        requires_categories=True,
        requires_keypoint_type=True,
    ),
    TaskType.OCR_BOX: TaskConfig(
        name="OCR Box",
        prompt_template="Detect all {categories} and recognize them.",
        description="Detect text in bounding boxes and recognize",
        output_format="boxes_with_text",
        requires_categories=True,
    ),
    TaskType.OCR_POLYGON: TaskConfig(
        name="OCR Polygon",
        prompt_template="Can you detect all {categories} in this image in polygon format like [x0, y0, x1, y1, x2, y2 ...] and then recognize them?",
        description="Detect text in polygons and recognize",
        output_format="polygons_with_text",
        requires_categories=True,
    ),
    TaskType.GUI_DETECTION: TaskConfig(
        name="GUI Detection",
        prompt_template='Detect element "{categories}"" in the image.',
        description="Detect GUI elements and return in bounding box format",
        output_format="boxes",
        requires_categories=True,
    ),
    TaskType.GUI_POINTING: TaskConfig(
        name="GUI Pointing",
        prompt_template='Point to element "{categories}".',
        description="Point to GUI elements and return in point format",
        output_format="points",
        requires_categories=True,
    ),
}


# Keypoint definitions for different object types
KEYPOINT_CONFIGS = {
    "person": [
        "nose",
        "left eye",
        "right eye",
        "left ear",
        "right ear",
        "left shoulder",
        "right shoulder",
        "left elbow",
        "right elbow",
        "left wrist",
        "right wrist",
        "left hip",
        "right hip",
        "left knee",
        "right knee",
        "left ankle",
        "right ankle",
    ],
    "animal": [
        "left eye",
        "right eye",
        "nose",
        "neck",
        "root of tail",
        "left shoulder",
        "left elbow",
        "left front paw",
        "right shoulder",
        "right elbow",
        "right front paw",
        "left hip",
        "left knee",
        "left back paw",
        "right hip",
        "right knee",
        "right back paw",
    ],
}


def get_task_config(task_type: TaskType) -> TaskConfig:
    """Get configuration for a task type"""
    return TASK_CONFIGS[task_type]


def get_keypoint_config(keypoint_type: str) -> Optional[List[str]]:
    """Get keypoint configuration for a specific type"""
    return KEYPOINT_CONFIGS.get(keypoint_type)
