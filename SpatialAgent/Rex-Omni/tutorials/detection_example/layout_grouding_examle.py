#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic object detection example using Rex Omni
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

    # Load image
    image_path = "tutorials/detection_example/test_images/layout.jpg"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")

    # Object detection
    categories = ["header", "headline", "paragraph", "page number", "figure", "section"]

    results = rex_model.inference(images=image, task="detection", categories=categories)

    # Print results
    result = results[0]
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
        output_path = "tutorials/detection_example/test_images/layout_visualize.jpg"
        vis_image.save(output_path)
        print(f"Visualization saved to: {output_path}")

    else:
        print(f"Inference failed: {result['error']}")


if __name__ == "__main__":
    main()
