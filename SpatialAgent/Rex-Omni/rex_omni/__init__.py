#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rex Omni: A high-level wrapper for Qwen2.5-VL multimodal language model
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .tasks import TaskType
from .utils import RexOmniVisualize
from .wrapper import RexOmniWrapper

__all__ = [
    "RexOmniWrapper",
    "TaskType",
    "RexOmniVisualize",
]
