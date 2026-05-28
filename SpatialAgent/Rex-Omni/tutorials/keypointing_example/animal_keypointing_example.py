"""
Animal keypointing example using Rex Omni
"""

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
    image_path = "tutorials/keypointing_example/test_images/animal.png"  # Replace with your image path
    image = Image.open(image_path).convert("RGB")
    print(f"✅ Image loaded successfully!")
    print(f"📏 Image size: {image.size}")

    # Animal keypointing
    print("🐱 Performing animal keypointing...")
    results = rex_model.inference(
        images=image, task="keypoint", keypoint_type="animal", categories=["cat"]
    )

    # Process results
    result = results[0]
    if result["success"]:
        predictions = result["extracted_predictions"]
        vis_image = RexOmniVisualize(
            image=image,
            predictions=predictions,
            font_size=6,
            draw_width=6,
            show_labels=True,
        )

        # Save visualization
        output_path = (
            "tutorials/keypointing_example/test_images/animal_keypointing_visualize.jpg"
        )
        vis_image.save(output_path)
        print(f"✅ Animal keypointing visualization saved to: {output_path}")

    else:
        print(f"❌ Inference failed: {result['error']}")


if __name__ == "__main__":
    main()
