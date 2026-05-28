#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GUI pointing example using Rex Omni
"""

import matplotlib.pyplot as plt
import torch
from PIL import Image

from rex_omni import RexOmniVisualize, RexOmniWrapper


def main():
    # Model path - replace with your actual model path
    model_path = "IDEA-Research/Rex-Omni"

    print("🚀 Initializing Rex Omni model...")

    # Create wrapper with custom parameters
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",  # Choose "transformers" or "vllm"
        max_tokens=2048,
        temperature=0.0,
        top_p=0.05,
        top_k=1,
        repetition_penalty=1.05,
    )

    # Load image
    image_path = "tutorials/detection_example/test_images/gui.png"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")
    print(f"✅ Image loaded successfully!")
    print(f"📏 Image size: {image.size}")

    # GUI pointing - find specific UI element
    categories = ["element 'pause current song'"]

    print("🖱️ Performing GUI pointing...")
    results = rex_model.inference(images=image, task="pointing", categories=categories)

    # Process results
    result = results[0]
    if result["success"]:
        predictions = result["extracted_predictions"]
        vis_image = RexOmniVisualize(
            image=image,
            predictions=predictions,
            font_size=50,
            draw_width=15,
            show_labels=True,
        )

        # Save visualization
        output_path = (
            "tutorials/pointing_example/test_images/gui_pointing_visualize.jpg"
        )
        vis_image.save(output_path)
        print(f"✅ GUI pointing visualization saved to: {output_path}")

    else:
        print(f"❌ Inference failed: {result['error']}")


if __name__ == "__main__":
    main()
