#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch Inference example using Rex-Omni
"""

import torch
from PIL import Image

from rex_omni import RexOmniVisualize, RexOmniWrapper


def main():
    # Model path - replace with your actual model path
    model_path = "IDEA-Research/Rex-Omni"

    # Create wrapper with custom parameters
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",  # or "vllm" for faster inference
        max_tokens=4096,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    # Load imag
    image_paths = [
        "tutorials/detection_example/test_images/cafe.jpg",
        "tutorials/detection_example/test_images/boys.jpg",
    ]
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    # Object detection
    categories = [
        [
            "man",
            "woman",
            "yellow flower",
            "sofa",
            "robot-shope light",
            "blanket",
            "microwave",
            "laptop",
            "cup",
            "white chair",
            "lamp",
        ],
        [
            "boys holding microphone",
            "boy playing piano",
            "the four guitars on the wall",
            "the guitar in someone's hand",
        ],
    ]

    results = rex_model.inference(
        images=images, task=["detection", "detection"], categories=categories
    )

    # Print results
    batch_idx = 0
    for result, image in zip(results, images):
        if result["success"]:
            predictions = result["extracted_predictions"]
            vis_image = RexOmniVisualize(
                image=image,
                predictions=predictions,
                font_size=20,
                draw_width=5,
                show_labels=True,
            )
            # Save visualization
            output_path = f"tutorials/other_example/batch_inference_{batch_idx}.jpg"
            vis_image.save(output_path)
            print(f"Visualization saved to: {output_path}")

        else:
            print(f"Inference failed: {result['error']}")
        batch_idx += 1


if __name__ == "__main__":
    main()
