import re
import os
import cv2
import ast
import torch
import argparse
import numpy as np
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
from utils.action_utils import image_to_base64, base64_to_image, expand_bbox
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor, AutoModelForZeroShotObjectDetection, Qwen2_5_VLForConditionalGeneration

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class BaseAction:
    """
        Base Action class for agent operations.
        This class serves as a foundation for wrapping APIs, tools, and models as actions that an agent can execute. 
    """

    def __init__(self, description: str = "", args_spec: dict = None, rets_spec: dict = None, examples: List = None) -> None:
        """
        Initialize a new action with metadata.
        
        Args:
            description (str): Human-readable description of what the action does
            args_spec (dict): Specification of the action's input arguments
            rets_spec (dict): Specification of the action's return values
            examples (List): List of example usages of the action
        """
        self.name = self.__class__.__name__
        self.description = description
        self.args_spec = args_spec or {}
        self.rets_spec = rets_spec or {}
        self.examples = examples or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, **kwargs) -> dict:
        raise NotImplementedError("Action implementations must override the __call__ method")


class Terminate(BaseAction):
    def __init__(self) -> None:
        description = """
            Use this function ONLY when you are completely confident in your final answer.
            For multiple-choice questions: Specify the letter of the correct option.
            For numerical answers: Include both the specific value and appropriate unit of measurement (e.g., meter or centimeter).
            For yes/no questions: Clearly state 'Yes' or 'No'.
            
            DO NOT call this function if you are uncertain or need to perform additional analysis.
            Double-check your answer before terminating!
        """
        args_spec = {"answer": "The final answer with proper formatting. For multiple choice: include letter (e.g., 'A. explanation' or '(B)'). For numerical answers: include units (e.g., '3.25 meters')."}
        rets_spec = {"answer": "The final answer that will be submitted."}
        examples = [
                {"name": "Terminate", "arguments": {"answer": "A. Yes."}},
                {"name": "Terminate", "arguments": {"answer": "(B)."}},
                {"name": "Terminate", "arguments": {"answer": "B. 3.25 meters."}},
                {"name": "Terminate", "arguments": {"answer": "(A) 2 inches."}},
                {"name": "Terminate", "arguments": {"answer": "47.3 centimeters."}},
                {"name": "Terminate", "arguments": {"answer": "38.2 degrees."}},
            ]

        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)
    
    def __call__(self, answer):
        if isinstance(answer, str) and answer.strip():
            return {"answer": answer}
        else:
            return {"answer": "No valid answer provided. Please provide a clear answer."}


class SelfThinking(BaseAction):
    def __init__(self, model=None, processor=None, late_load=True):
        description = """
            Modes:
            1. Text-only: Provide 'query' for pure language tasks.
            2. Vision+Language: Provide 'images' + 'query' for visual analysis.

            Suitable for: Scene understanding, OCR, object/color recognition, classification, and concept-level Q&A.
        """

        args_spec = {
            "query": "Text question or instruction (REQUIRED).",
            "image": "List of image paths. If omitted, the model performs text-only reasoning.",
        }

        rets_spec = {"response": "Model's response string."}

        examples = [
            {"name": "SelfThinking", "arguments": {"query": "Summarize the image content.", "image": ["image-0"]}}
        ]

        super().__init__(description, args_spec, rets_spec, examples)

        if model and processor:
            self.model, self.processor = model, processor
        elif not late_load:
            self._initialize_model()
        else:
            self.model, self.processor = None, None

        self.device = next(self.model.parameters()).device if self.model else "cuda"

    def _initialize_model(self):
        model_path = "/remote-home/share/huggingface/Qwen3-VL-8B-Instruct"

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=2560 * 28 * 28,
        )

        from transformers import Qwen3VLForConditionalGeneration
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,attn_implementation="flash_attention_2")
        self.model.eval()
        print("Qwen3-VL-8B-Instruct loaded successfully")

    def __call__(self, query, image=None):
        if self.model is None or self.processor is None:
            print("[SelfReasoning] Model not initialized — loading Qwen3-VL-8B-Instruct now...")
            self._initialize_model()

        assistant_prompt = (
            "You are a helpful assistant specializing in visual and textual analysis. "
            "If an image is provided, describe or answer based on what is visible. "
            "If no image is provided, answer using your general knowledge."
        )

        # ---- Text-only mode ----
        if not image or len(image) == 0:
            messages = [
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": query},
            ]
            input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[input_text], padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=256, use_cache=True, do_sample=False, temperature=0.0)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
                response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            torch.cuda.empty_cache()
            
            return {"response": response}

        # ---- Vision+Language mode ----
        image_pils = [Image.open(img).convert("RGB") for img in image]
        image_contents = [{"type": "image", "image": img_path} for img_path in image]

        messages = [
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": image_contents + [{"type": "text", "text": query}]},
        ]
        input_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[input_text], images=image_pils, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, use_cache=True, do_sample=False, temperature=0.0)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        torch.cuda.empty_cache()

        return {"response": response}


class LocalizeObjects(BaseAction):
    def __init__(self):
        description = """Localize specific objects in an image. Returns bounding boxes for target categories, optionally visualizing them."""
        args_spec = {
            "image": "The image to analyze.",
            "objects": "A list of object categories to detect.",
            "visualize_output": "Whether to draw and save visualization image."
        }
        rets_spec = {"regions": "List of detected regions with label, bbox"}
        examples = [{
            "name": "LocalizeObjects",
            "arguments": {"image": "image-0", "objects": ["dog", "cat"], "visualize_output": True}
        }]
        super().__init__(description, args_spec, rets_spec, examples)

        from rex_omni import RexOmniWrapper
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = RexOmniWrapper(model_path="/remote-home/share/huggingface/Rex-Omni", backend="transformers")

    def __call__(self, image, objects: List[str], visualize_output: bool = False):
        from PIL import Image as PILImage, ImageDraw

        # 读取图片
        if isinstance(image, str):
            image_pil = PILImage.open(image).convert("RGB")
        else:
            image_pil = image.convert("RGB")

        results = self.model.inference(images=image_pil, task="detection", categories=objects)

        if not results or not results[0].get("success", False):
            return {"regions": []}

        predictions = results[0]["extracted_predictions"]
        regions = []

        if visualize_output:
            draw = ImageDraw.Draw(image_pil)

        for label, preds in predictions.items():
            for pred in preds:
                if pred["type"] != "box":
                    continue
                bbox = pred["coords"]  # 假设是像素坐标
                bbox = [float(x) for x in bbox]
                regions.append({"label": label, "bbox": bbox})

                if visualize_output:
                    draw.rectangle(bbox, outline="red", width=3)
                    draw.text((bbox[0], bbox[1] - 10), label, fill="red")

        if visualize_output:
            vis_path = "./output/localized_objects_vis.png"
            image_pil.save(vis_path)
            print(f"Visualization saved to {vis_path}")

        return {"regions": regions}


class CountObjects(BaseAction):
    def __init__(self):        
        description = """Count target objects in an image. Returns the coordinates of each detected target as points."""
        args_spec = {"image": "The image to analyze.", "objects": "List of object categories to count."}
        rets_spec = {"points": "Dictionary {category: [points...]}, points in normalized coordinates."}
        examples = [{"name": "CountObjects", "arguments": {"image": "image-0", "objects": ["bed"]}}]
        super().__init__(description, args_spec, rets_spec, examples)

        from rex_omni import RexOmniWrapper

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model = RexOmniWrapper(model_path="/remote-home/share/huggingface/Rex-Omni", backend="transformers")

    def __call__(self, image, objects: List[str]):
        from PIL import Image as PILImage
        if isinstance(image, str):
            image_pil = PILImage.open(image).convert("RGB")
        else:
            image_pil = image.convert("RGB")

        results = self.model.inference(images=image_pil, task="pointing", categories=objects)

        if not results or not results[0].get("success", False):
            return {"points": {}}

        predictions = results[0]["extracted_predictions"]  # dict: {label: [{type:'point', coords: [...]}]}
        
        return {"points": predictions}


class GetObjectMask(BaseAction):
    def __init__(self) -> None:
        description = """
            Generate pixel-level segmentation masks for specified objects.
            Returns mask area ratios and bounding boxes for each detected object.
            Suitable for analyzing object shapes, sizes, and coverage.
        """

        args_spec = {"image": "Image file to process.", "objects": "List of object descriptions to localize and segment"}

        rets_spec = {
            "results": "List of dicts with mask area ratio, bounding box, and optional error: [{'object': str, 'mask_area': float, 'bbox': [left, top, right, bottom], 'error': str or None}]"
        }

        examples = [{"name": "GetObjectMask", "arguments": {"image": "image-0", "objects": ["coffee mug", "microwave"]}}]
        
        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_config = "configs/sam2/sam2_hiera_l.yaml" 
        # Here is a bug of hiera, can only use relative path of build_sam
        # haoning: need to install iopath to avoid this error
        
        self.predictor = SAM2ImagePredictor(build_sam2(model_config, checkpoint))
    
    def create_combined_overlay(self, original_img, masks_data):
        """
        Create a single overlay with all object masks on the original image.
        
        Returns:
            PIL Image with all masks overlaid using different colors
        """
        # Create a copy of the original image for overlay
        result = original_img.copy().convert("RGBA")
        
        # Predefined colors for different objects: Red, Green, Blue, Yellow, Magenta, Cyan, Orange, Purple, Light Blue, Pink
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (255, 0, 128)]
        draw, font = ImageDraw.Draw(result), ImageFont.load_default()
        
        # Add each mask to the combined image
        for i, mask_data in enumerate(masks_data):
            if "error" in mask_data and mask_data["error"]:
                continue
                
            mask = mask_data.get("mask")
            label = mask_data.get("object")
            
            if mask is None:
                continue
                
            color = colors[i % len(colors)]  # Cycle through colors
            alpha = 0.5  # Transparency
            
            # Convert mask to RGBA for overlay
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for c in range(3):
                mask_rgb[:, :, c] = mask * color[c]
                
            mask_overlay = Image.fromarray(mask_rgb).convert("RGBA")
            mask_overlay.putalpha(int(alpha * 255))
            
            # Composite the mask onto the result
            result.paste(mask_overlay, (0, 0), mask_overlay)
            
            # Add label if provided
            if label:
                # Find a good position for the label (top of the mask)
                mask_indices = np.where(mask)
                if len(mask_indices[0]) > 0:
                    y_min = np.min(mask_indices[0])
                    x_position = np.mean(mask_indices[1][mask_indices[0] == y_min])
                    
                    # Draw white text with black outline
                    draw.text((x_position, y_min - 15), label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        
        return result.convert("RGB")  # Convert back to RGB for consistency
    
    def __call__(self, image: str, objects: List[str], visualize_output=False) -> dict:
        """
        Generate masks for multiple objects and return their area ratios and bounding boxes.

        Returns:
            dict: {"results": [{"object": str, "mask_area": float, "bbox": [left, top, right, bottom], "error": str or None}]}
        """

        image = Image.open(image).convert("RGB")
        H, W = np.array(image).shape[:2]
        total_pixels = H * W

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)

        # Get bounding boxes from LocalizeObjects
        localize_action = LocalizeObjects()
        regions = localize_action(image, objects)["regions"]
        region_dict = {region["label"]: region for region in regions}

        results, masks_data = [], []
        for obj in objects:
            result = {"object": obj, "mask_area": 0.0, "bbox": [0.0, 0.0, 0.0, 0.0], "error": None}
            if obj in region_dict:
                region = region_dict[obj]
                bbox = region["bbox"]
                
                with torch.no_grad():
                    masks, scores, _ = self.predictor.predict(box=np.array(bbox), multimask_output=False)
                mask = masks[0]
                
                result["mask_area"] = float(np.sum(mask) / total_pixels)
                result["bbox"] = [round(float(x), 4) for x in bbox]

                # Convert binary mask to PIL image (0 for background, 255 for foreground)
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                result["mask_image"] = image_to_base64(mask_image)

                # Store mask for combined overlay
                masks_data.append({"object": obj, "mask": mask})
            else:
                result["error"] = "Object not detected."
            results.append(result)

        # Create combined overlay with all detected objects
        if visualize_output:
            combined_overlay = self.create_combined_overlay(image, masks_data)
            combined_overlay.save("combined_overlay.png")  # Save the combined overlay image

        return {"results": results}


class Detect3DObjects(BaseAction):
    def __init__(self):
        description = """
            Detect specific objects in an image and estimate their 3D bounding boxes.
            Returns 3D bounding box parameters in the following format:

                x, y, z -> object center in camera coordinates (meters);
                width, height, length -> physical size (width, height, length) in meters;
                yaw -> heading angle around vertical axis (radians)
        """
        args_spec = {"image": "Path to the input image.", "objects": "List of object categories to detect (or a single string)."}
        rets_spec = {"objects": "List of dicts with {label: str, bbox_3d: {x:float, y:float, z:float, width:float, height:float, length:float, yaw:float}}"}
        examples = [
            {"name": "Detect3DObjects", "arguments": {"image": "image-1", "objects": ["dog", "rabbit"]}},
        ]
        super().__init__(description, args_spec, rets_spec, examples)

    def __call__(self, image, objects, visualize_output=False):
        from DetAny3D.spatial_tool import predict
        if isinstance(objects, str):
            objects = [objects]
        text_prompt = ", ".join(objects)

        outputs = predict(image, text_prompt)

        if visualize_output and "image" in outputs:
            os.makedirs("./output", exist_ok=True)
            save_path = "./output/detect3d_vis.jpg"
            cv2.imwrite(save_path, cv2.cvtColor(outputs["image"], cv2.COLOR_RGB2BGR))
            print(f"[Detect3DObjects] Visualization saved to {save_path}")

        return {"objects": outputs["objects"]}


class EstimateImageDepth(BaseAction):
    def __init__(self) -> None:
        description = """
            Estimate the depth map (in meters) of an entire image using the DepthAnythingV2 with metric depth estimation.
            Supports both indoor (max depth 20m) and outdoor scenes (max depth 80m) by selecting the appropriate pre-loaded model based on the 'indoor_or_outdoor' parameter.
            Returns a visualization of the depth map as a base64-encoded image.
            ALL DEPTH VALUES ARE RETURNED IN METERS, not in arbitrary units or pixels.
        """
        args_spec = {
            "image": "The image to analyze.",
            "indoor_or_outdoor": "Scene type ('indoor' or 'outdoor') to select the appropriate depth model."
        }
        rets_spec = {"depth_map": "The estimated depth map (in meters) of the image as a base64-encoded image."}
        examples = [
            {"name": "EstimateImageDepth", "arguments": {"image": "image-0", "indoor_or_outdoor": "indoor"}},
            {"name": "EstimateImageDepth", "arguments": {"image": "image-1", "indoor_or_outdoor": "outdoor"}},
        ]
        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)

        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        self.encoder = 'vitl'  # Can be changed to 'vits' or 'vitb' for different model sizes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Indoor and Outdoor model configurations
        self.indoor_config = {'dataset': 'hypersim', 'max_depth': 20, 'checkpoint_path': "./DepthAnythingV2/ckpt/hypersim.pth"}
        self.outdoor_config = {'dataset': 'vkitti', 'max_depth': 80, 'checkpoint_path': "./DepthAnythingV2/ckpt/vkitti.pth"}

        from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        # Indoor Model
        self.indoor_model = DepthAnythingV2(**{**self.model_configs[self.encoder], 'max_depth': self.indoor_config['max_depth']})
        self.indoor_model.load_state_dict(torch.load(self.indoor_config['checkpoint_path'], map_location='cpu'))
        self.indoor_model.eval()
        self.indoor_model.to(self.device)

        # Outdoor Model
        self.outdoor_model = DepthAnythingV2(**{**self.model_configs[self.encoder], 'max_depth': self.outdoor_config['max_depth']})
        self.outdoor_model.load_state_dict(torch.load(self.outdoor_config['checkpoint_path'], map_location='cpu'))
        self.outdoor_model.eval()
        self.outdoor_model.to(self.device)
        
        self._depth_cache = {}  # Cache for depth maps to avoid recomputing

    def get_depth_map(self, image, indoor_or_outdoor="indoor"):
        # Create a cache key based on image path and scene type
        if isinstance(image, str):
            cache_key = f"{image}_{indoor_or_outdoor}"
        else:
            cache_key = None # For PIL images, we can't cache effectively
            
        # Return cached result if available
        if cache_key and cache_key in self._depth_cache:
            return self._depth_cache[cache_key]
            
        # Select appropriate model
        if indoor_or_outdoor.lower() == "indoor":
            model = self.indoor_model
        elif indoor_or_outdoor.lower() == "outdoor":
            model = self.outdoor_model
        else:
            raise ValueError(f"Invalid scene type: {indoor_or_outdoor}. Must be 'indoor' or 'outdoor'")
            
        # Load image if needed
        if isinstance(image, str):
            raw_img = cv2.imread(image)
        else:
            # Convert PIL Image to cv2 format
            raw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        # Generate depth map
        with torch.no_grad():
            depth_map = model.infer_image(raw_img)  # Returns HxW depth map in meters (numpy array)
            
        # Cache result if possible
        if cache_key:
            self._depth_cache[cache_key] = depth_map
            
        return depth_map
        
    def get_region_depth(self, depth_map, bbox, mode="mean"):
        """
        Extract depth for a specific region from a depth map
        
        Returns:
            float: Estimated depth value
        """
        H, W = depth_map.shape
        
        # Validate and adjust bbox to ensure it's within image bounds
        x1, y1, x2, y2 = map(int, bbox)  # Convert to pixel coordinates
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1, min(x2, W - 1))
        y2 = max(y1, min(y2, H - 1))
        
        if mode == "center":
            x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
            depth_value = depth_map[y_center, x_center]
        elif mode == "mean":
            depth_region = depth_map[y1:y2, x1:x2]
            depth_value = np.mean(depth_region)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'mean' or 'center'")
            
        return depth_value

    def __call__(self, image: str, indoor_or_outdoor: str = "indoor") -> Dict:
        """
        Get the depth map for a whole image.

        Returns:
            dict: {"depth_map": numpy array} - Complete depth map of the image
        """
        depth_map = self.get_depth_map(image, indoor_or_outdoor)
        
        # Create a visualization of the depth map
        # Normalize depth map for better visualization
        depth_vis = depth_map.copy()
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min()) * 255
        depth_vis = depth_vis.astype(np.uint8)
        
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        # Convert to PIL Image for base64 encoding
        depth_image = Image.fromarray(depth_colored)

        torch.cuda.empty_cache()  # Clear GPU memory

        return {"depth_map": image_to_base64(depth_image)}


class EstimateRegionDepth(BaseAction):
    def __init__(self, depth_estimator=None) -> None:
        description = """
            Estimate metric depth (in meters) of specified regions in an image.
            Supports indoor (0–20m) and outdoor (0–80m) scenes.
            Works with single or multiple bounding boxes in pixel coordinates.
            Depth is distance from camera to object, not between objects or object size.
        """

        args_spec = {
            "image": "Image to analyze.",
            "bboxes": "Bounding box or list of boxes in pixel coordinates: [left, top, right, bottom] or [[...], ...].",
            "indoor_or_outdoor": "Scene type ('indoor' or 'outdoor').",
            "mode": "Depth calculation: 'mean' (average) or 'center' (center point). Default: 'mean'."
        }

        rets_spec = {
            "depths": "List of dicts with bbox, depth (m), and optional error: [{'bbox': list, 'depth': float, 'error': str or None}]",
            "unit": "Always 'meters'."
        }

        examples = [
            {"name": "EstimateRegionDepth", "arguments": {"image": "image-0", "bboxes": [[100, 50, 200, 150], [150, 100, 250, 200]], "indoor_or_outdoor": "indoor"}}
        ]

        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)
        # Use the provided depth estimator or create a new one
        self.depth_estimator = depth_estimator or EstimateImageDepth()

    def __call__(self, image: str, bboxes, indoor_or_outdoor: str = "indoor", mode: str = "mean") -> Dict:
        """
        Estimate the depth of a region in the image using the depth estimator.

        Returns:
            dict: {"depth": float} - Estimated depth in meters, rounded to 2 decimal places.
        """
        # Get depth map for the entire image
        depth_map = self.depth_estimator.get_depth_map(image, indoor_or_outdoor)
        
        # Normalize input to always be a list of bounding boxes
        if not isinstance(bboxes, list):
            return {"depths": [], "unit": "meters", "error": "bboxes must be a list"}
            
        if len(bboxes) > 0 and not isinstance(bboxes[0], list):
            # Single bbox case: [x1, y1, x2, y2]
            bboxes = [bboxes]
        
        # Extract depth for the specific region
        depths = []
        for bbox in bboxes:
            try:
                depth_value = self.depth_estimator.get_region_depth(depth_map, bbox, mode)
                depths.append({"bbox": bbox, "depth": round(float(depth_value), 2), "error": None})
            except Exception as e:
                depths.append({"bbox": bbox, "depth": None, "error": str(e)})
         
        torch.cuda.empty_cache()  # Clear GPU memory

        return {"depths": depths, "unit": "meters"}


class EstimateObjectDepth(BaseAction):
    def __init__(self, object_localizer=None, depth_estimator=None) -> None:
        description = """
            Estimate object depth (in meters) from an image.
            Supports indoor (0–20m) and outdoor (0–80m) scenes.
            Depth indicates distance from camera to object, not between objects or object size.
        """

        args_spec = {
            "image": "Image to analyze.",
            "objects": "List of object descriptions to measure distance to (e.g., ['dog', 'cat']).",
            "indoor_or_outdoor": "Scene type ('indoor' or 'outdoor')."
        }

        rets_spec = {"results": "List of dicts with object description, depth (m), and optional error: [{'object': str, 'depth': float, 'error': str or None}]"}

        examples = [{"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "objects": ["the red car", "dog"], "indoor_or_outdoor": "outdoor"}}]

        
        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)
        
        # Use provided instances or create new ones
        self.object_localizer = object_localizer or LocalizeObjects()
        # For best efficiency, use the shared depth estimator
        self.image_depth_estimator = depth_estimator if isinstance(depth_estimator, EstimateImageDepth) else EstimateImageDepth()
    
    def __call__(self, image: str, objects: List[str], indoor_or_outdoor: str = "indoor", mode: str = "mean") -> dict:
        """
            Estimate the depth of multiple objects in an image.

            Returns:
                dict: {"results": [{"object": str, "depth": float or str, "error": str or None}]}
        """
        # Get object regions
        regions = self.object_localizer(image=image, objects=objects)["regions"]
        region_dict = {region["label"]: region for region in regions}
        
        # Get the depth map once for the entire image (key optimization)
        depth_map = self.image_depth_estimator.get_depth_map(image, indoor_or_outdoor)

        results = []
        for obj in objects:
            result = {"object": obj, "depth": None, "error": None}
            if obj in region_dict:
                bbox = region_dict[obj]["bbox"]
                depth = self.image_depth_estimator.get_region_depth(depth_map, bbox, mode)
                
                if depth > 0:
                    result["depth"] = round(float(depth), 3)
                else:
                    result["error"] = "Invalid depth value."
            else:
                result["error"] = "Object not detected."
            results.append(result)

        torch.cuda.empty_cache()  # Clear GPU memory

        return {"results": results}


class GetCameraParametersVGGT(BaseAction):
    def __init__(self) -> None:
        description = """
            Extract camera extrinsic (3*4, relative to first image) and intrinsic (3*3) parameters from images using VGGT.
            Useful for 3D reconstruction, novel view synthesis, and geometric analysis.
        """

        args_spec = {
            "image": "List of image paths (at least one).",
            "dtype": "Torch data type: 'bfloat16' (Ampere GPUs) or 'float16' (default auto-select)."
        }

        rets_spec = {"output": "List of dicts with image_index (int), extrinsic (3*4 matrix), and intrinsic (3*3 matrix)."}

        examples = [{"name": "GetCameraParametersVGGT", "arguments": {"image": ["image-0", "image-1"], "dtype": "float16"}}]
        
        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Select dtype based on GPU capability (bfloat16 for Compute Capability 8.0+)
        self.default_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        from vggt.models.vggt import VGGT
        self.model = VGGT.from_pretrained("/remote-home/share/huggingface/VGGT-1B")
        self.model = self.model.to(self.device).eval()

    def __call__(self, image: List[str], dtype: str = "auto") -> Dict:
        if not image:
            raise ValueError("At least one image path is required.")

        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        
        # Determine inference data type
        if dtype == "auto":
            torch_dtype = self.default_dtype
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            raise ValueError("dtype must be 'auto', 'bfloat16', or 'float16'.")

        # Load and preprocess images
        img_tensor = load_and_preprocess_images(image).to(self.device)
        
        # Run VGGT inference
        with torch.no_grad():
            with torch.amp.autocast(dtype=torch_dtype, device_type=self.device):
                predictions = self.model(img_tensor)
                pose_enc = predictions['pose_enc']
                extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, img_tensor.shape[-2:])
        
        # Process results into the expected output format
        output_list = []
        for batch_idx in range(extrinsics.shape[0]): # Batch
            for img_idx in range(extrinsics.shape[1]):
                output_list.append({
                    "image_index": img_idx,
                    "extrinsic": extrinsics[batch_idx, img_idx].cpu().tolist(),
                    "intrinsic": intrinsics[batch_idx, img_idx].cpu().tolist()
                })
        torch.cuda.empty_cache()  # Clear GPU memory

        return {"output": output_list}


class EstimateObjectGeometryProperties(BaseAction):
    def __init__(self, mask_generator=None, depth_estimator=None, camera_params_estimator=None) -> None:
        description = """
            Analyze objects in an image to obtain bounding boxes, mask areas, depth (m), and camera parameters.
            Camera parameters include intrinsic (3*3) and extrinsic (3*4) matrices for 3D geometry tasks.
        """

        args_spec = {"image": "Image file path to analyze.", "object_descs": "List of object descriptions (e.g., ['dog', 'cat'])."}

        rets_spec = {
            "results": "List of dicts with object, bbox, mask_area, depth (m), and optional error.",
            "camera_parameters": "Dict with intrinsic (3*3) and extrinsic (3*4) matrices."
        }

        examples = [{"name": "EstimateObjectGeometryProperties", "arguments": {"image": "image-0", "object_descs": ["coffee cup", "keyboard"]}}]

        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)

        self.mask_generator = mask_generator or GetObjectMask()
        self.depth_estimator = depth_estimator or EstimateObjectDepth()
        self.camera_params_estimator = camera_params_estimator or GetCameraParametersVGGT()
    
    def __call__(self, image: str, object_descs: List[str]) -> dict:
        """
            Estimate the real-world dimensions of multiple objects by calculating their physical size based on mask, depth, and camera parameters.

            Returns:
                dict: {"sizes": [{"object": str, "width": float, "height": float, "area": float, "depth": float, "mask_area": float, "error": str or None}]}
        """

        results, camera_parameters = [], {"intrinsic": [], "extrinsic": []}

        # Step 1: Get camera parameters (call once for the entire image)
        try:
            camera_result = self.camera_params_estimator(images=[image])
            camera_output = camera_result.get("output", [])
            if camera_output and len(camera_output) > 0:
                camera_parameters["intrinsic"] = camera_output[0].get("intrinsic", [])
                camera_parameters["extrinsic"] = camera_output[0].get("extrinsic", [])
                # Validate camera parameters
                if not (isinstance(camera_parameters["intrinsic"], list) and len(camera_parameters["intrinsic"]) == 3 and
                        all(len(row) == 3 for row in camera_parameters["intrinsic"])):
                    raise ValueError("Invalid intrinsic matrix format.")
                if not (isinstance(camera_parameters["extrinsic"], list) and len(camera_parameters["extrinsic"]) == 3 and
                        all(len(row) == 4 for row in camera_parameters["extrinsic"])):
                    raise ValueError("Invalid extrinsic matrix format.")
            else:
                raise ValueError("No camera parameters returned.")
        except Exception as e:
            print(f"Warning: Failed to get camera parameters, returning empty matrices: {str(e)}")
            camera_parameters = {"intrinsic": [], "extrinsic": []}

        # Step 2: Get mask areas and bounding boxes
        mask_result = self.mask_generator(image=image, objects=object_descs)
        mask_data = mask_result.get("results", [])

        # Step 3: Get depths for objects
        depth_result = self.depth_estimator(image=image, objects=object_descs, indoor_or_outdoor="outdoor")
        depth_data = depth_result.get("results", [])

        # Create dictionaries for quick lookup
        mask_dict = {item["object"]: item for item in mask_data}
        depth_dict = {item["object"]: item for item in depth_data}

        # Process each object
        for obj_desc in object_descs:
            result = {"object": obj_desc, "bbox": [0.0, 0.0, 0.0, 0.0], "mask_area": 0.0, "depth": 0.0, "error": None}
            # Check if object exists in both mask and depth results
            if obj_desc not in mask_dict or obj_desc not in depth_dict:
                result["error"] = "Object not detected or failed to process."
                results.append(result)
                continue

            mask_item, depth_item = mask_dict[obj_desc], depth_dict[obj_desc]
            mask_area, bbox, depth = mask_item.get("mask_area"), mask_item.get("bbox"), depth_item.get("depth")

            # Validate inputs
            try:
                if mask_item.get("error") or not isinstance(mask_area, (int, float)) or mask_area < 0:
                    raise ValueError(f"Invalid mask_area for {obj_desc}: {mask_area}")
                if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox)):
                    raise ValueError(f"Invalid bbox for {obj_desc}: {bbox}")
                if depth_item.get("error") or not isinstance(depth, (int, float)) or depth <= 0:
                    raise ValueError(f"Invalid depth for {obj_desc}: {depth}")
            except Exception as e:
                result["error"] = f"Input validation failed: {str(e)}"
                results.append(result)
                continue

            # Assign validated data
            result["mask_area"] = float(mask_area)
            result["bbox"] = [float(x) for x in bbox]
            result["depth"] = float(depth)
            
            # Calculate additional metrics if needed (could be expanded later)
            x1, y1, x2, y2 = bbox
            result["width_pixels"] = x2 - x1
            result["height_pixels"] = y2 - y1
            
            results.append(result)

        return {"results": results, "camera_parameters": camera_parameters}


class MatchImagesSIFT(BaseAction):
    def __init__(self) -> None:
        description = """
            Match keypoints between two images using SIFT.
            Detects distinctive features and returns matched coordinate pairs for tasks like alignment or recognition.
        """

        args_spec = {
            "image": "List of two image paths.",
            "num_keypoints": "Max keypoints per image (default: 1200).",
            "ratio_th": "Ratio test threshold for matching (default: 0.75)."
        }

        rets_spec = {"matches": "List of matched coordinate pairs: [[x1, y1], [x2, y2]].", "num_matches": "Total number of matches found."}

        examples = [{"name": "MatchImagesSIFT", "arguments": {"image": ["image-0", "image-1"], "num_keypoints": 1200, "ratio_th": 0.75}}]


        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)

    def __call__(self, image: List[str], num_keypoints: int = 1200, ratio_th: float = 0.75, visualize_output=False) -> Dict:
        if len(image) != 2:
            raise ValueError("Exactly 2 images required.")

        img1, img2 = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE), cv2.imread(image[1], cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise ValueError("Failed to load images.")

        self.sift = cv2.SIFT_create(nfeatures=num_keypoints)

        # Detect keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        # Handle edge cases with empty keypoints
        if not kp1 or not kp2 or des1 is None or des2 is None:
            return {"matches": [], "num_matches": 0}

        # Convert keypoint coordinates to arrays
        kpts1 = np.array([kp.pt for kp in kp1], dtype=np.float32)
        kpts2 = np.array([kp.pt for kp in kp2], dtype=np.float32)
        
        des1, des2 = des1.astype(np.float32), des2.astype(np.float32)
        
        # Match keypoints
        bf = cv2.BFMatcher()
        matched_pairs = []
        
        if len(kp1) > 0 and len(kp2) > 0:
            knn_matches = bf.knnMatch(des1, des2, k=2)
            # Filter matches using ratio test
            for m, n in knn_matches:
                if m.distance < ratio_th * n.distance:
                    pt1 = [round(float(kpts1[m.queryIdx][0]), 2), round(float(kpts1[m.queryIdx][1]), 2)]
                    pt2 = [round(float(kpts2[m.trainIdx][0]), 2), round(float(kpts2[m.trainIdx][1]), 2)]
                    matched_pairs.append([pt1, pt2])

        # Save visualization
        if visualize_output:
            vis_path = os.path.join("./sift-matches.jpg")

            if matched_pairs:
                # Create visualization of matches
                kp1_matched = [cv2.KeyPoint(x[0], x[1], 1) for x, _ in matched_pairs]
                kp2_matched = [cv2.KeyPoint(x[0], x[1], 1) for _, x in matched_pairs]
                good_matches = [cv2.DMatch(i, i, 0) for i in range(len(matched_pairs))]
                
                vis_img = cv2.drawMatches(
                    img1, kp1_matched, img2, kp2_matched, good_matches, None,
                    matchColor=(255, 0, 0), singlePointColor=(0, 0, 255), flags=2
                )
                cv2.imwrite(vis_path, vis_img)

        return {"matches": matched_pairs, "num_matches": len(matched_pairs)}


class EstimateHomographyMatrix(BaseAction):
    def __init__(self) -> None:
        description = """
            Compute a 3*3 homography matrix between two images using SIFT features and RANSAC.
            Useful for alignment, perspective correction, and planar transformations.
        """

        args_spec = {
            "image": "List of two image paths.",
            "num_keypoints": "Max keypoints per image (default: 1200).",
            "ratio_th": "Ratio test threshold (default: 0.75).",
            "ransac_reproj_threshold": "Max reprojection error in RANSAC (default: 5.0)."
        }

        rets_spec = {
            "homography_matrix": "3*3 matrix mapping points from first image to second.",
            "inliers_count": "Number of inlier matches used.",
            "total_matches": "Total matches found.",
            "status": "Success or failure."
        }

        examples = [
            {"name": "EstimateHomographyMatrix", "arguments": {"image": ["image-0", "image-1"], "num_keypoints": 1200, "ratio_th": 0.75, "ransac_reproj_threshold": 5.0}}
        ]

        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)

    def __call__(self, image: List[str], num_keypoints: int = 1200, ratio_th: float = 0.75, ransac_reproj_threshold: float = 5.0, visualize_output=False) -> Dict:
        if len(image) != 2:
            raise ValueError("Exactly 2 images required.")

        img1, img2 = cv2.imread(image[0], cv2.IMREAD_GRAYSCALE), cv2.imread(image[1], cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            raise ValueError("Failed to load images.")

        self.sift = cv2.SIFT_create(nfeatures=num_keypoints)

        # Detect keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        # Handle edge cases with empty keypoints
        if not kp1 or not kp2 or des1 is None or des2 is None:
            return {"homography_matrix": None, "inliers_count": 0, "total_matches": 0, "status": "failure"}

        # Convert keypoint coordinates to arrays
        kpts1 = np.array([kp.pt for kp in kp1], dtype=np.float32)
        kpts2 = np.array([kp.pt for kp in kp2], dtype=np.float32)
        
        des1, des2 = des1.astype(np.float32), des2.astype(np.float32)
        
        # Match keypoints
        bf = cv2.BFMatcher()
        matched_pairs, src_pts, dst_pts = [], [], []
        
        if len(kp1) > 0 and len(kp2) > 0:
            knn_matches = bf.knnMatch(des1, des2, k=2)
            # Filter matches using ratio test
            for m, n in knn_matches:
                if m.distance < ratio_th * n.distance:
                    pt1 = kpts1[m.queryIdx]
                    pt2 = kpts2[m.trainIdx]
                    matched_pairs.append([pt1.tolist(), pt2.tolist()])
                    src_pts.append(pt1)
                    dst_pts.append(pt2)
        
        total_matches = len(matched_pairs)
        
        # Compute homography using RANSAC if enough matches
        if len(src_pts) >= 4:
            src_pts, dst_pts = np.array(src_pts), np.array(dst_pts)
            
            # Find homography matrix
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
            inliers_count = int(np.sum(mask))
            
            # Save visualization if required
            if visualize_output:
                vis_path = "./homography-visualization.jpg"
                
                # Create a visualization showing the inliers
                matchesMask = mask.ravel().tolist()
                
                # Convert to KeyPoint and DMatch objects for visualization
                kp1_vis = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in src_pts]
                kp2_vis = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in dst_pts]
                good_matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
                
                draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
                
                img_matches = cv2.drawMatches(img1, kp1_vis, img2, kp2_vis, good_matches, None, **draw_params)
                cv2.imwrite(vis_path, img_matches)
            
            # Convert homography matrix to list for output
            homography_matrix = H.tolist() if H is not None else None
            status = "success" if H is not None else "failure"
            
            return {"homography_matrix": homography_matrix, "inliers_count": inliers_count, "total_matches": total_matches, "status": status}
        else:
            # Not enough matches to compute homography
            return {"homography_matrix": None, "inliers_count": 0, "total_matches": total_matches, "status": "insufficient_matches"}


class EstimateOpticalFlow(BaseAction):
    def __init__(self) -> None:
        description = """
            Estimate optical flow between two images to measure motion in pixels.
            Returns average displacement in horizontal (x) and vertical (y) directions.

            First image is earlier in time; second is later.
            - mean_flow_x > 0: objects move left / camera moves right.
            - mean_flow_x < 0: objects move right / camera moves left.
            - mean_flow_y > 0: objects move up / camera moves down.
            - mean_flow_y < 0: objects move down / camera moves up.

            Useful for analyzing camera motion, object movement, and 3D spatial reasoning.
        """

        args_spec = {"image": "A list of exactly two image paths to compute optical flow between. First image is earlier in time."}
        rets_spec = {"output": "Dictionary containing 'mean_flow_x' (average horizontal pixel displacement) and 'mean_flow_y' (average vertical pixel displacement)."}
        examples = [
            {
                "name": "EstimateOpticalFlow", "arguments": {"image": ["image-1", "image-3"]}
            }
        ]

        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)

        from RAFT.core.raft import RAFT

        model_path = "./checkpoints/RAFT/raft-things.pth"
        args = argparse.Namespace(model=model_path, path="demo-frames", small=False, mixed_precision=False, alternate_corr=False)
        
        self.model = RAFT(args)
        state_dict = torch.load(model_path)
        
        if any(k.startswith("module.") for k in state_dict.keys()): # Handle module. prefix compatibility
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_image(self, imfile: str) -> torch.Tensor:
        # Returns: torch.Tensor: Image tensor [1, C, H, W].
        if not os.path.exists(imfile):
            raise ValueError(f"Image file does not exist: {imfile}")
        img = np.array(Image.open(imfile)).astype(np.uint8)
        if img.ndim == 2:  # Convert grayscale to RGB
            img = np.stack([img] * 3, axis=-1)
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img[None].to(self.device)

    def __call__(self, image: List[str]) -> Dict:
        """
        Compute average optical flow between two images.
        
        Returns:
            Dict: Dictionary with 'output' containing 'mean_flow_x' and 'mean_flow_y'.
        """
        from RAFT.core.utils.utils import InputPadder
        if len(image) != 2:
            raise ValueError(f"Exactly 2 images required, got {len(image)}: {image}")

        # Load images
        image1 = self.load_image(image[0]) # Earlier frame
        image2 = self.load_image(image[1]) # Later frame

        # Pad images to ensure dimensions are divisible by 8
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        # Compute optical flow
        with torch.no_grad():
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)

        # Extract flow field and compute average displacement
        flow = flow_up[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        mean_flow_x = np.mean(flow[..., 0])  # Average x-component (positive: left, negative: right)
        mean_flow_y = np.mean(flow[..., 1])  # Average y-component (positive: up, negative: down)

        return {"output": {"mean_flow_x": float(mean_flow_x), "mean_flow_y": float(mean_flow_y)}}


class GetObjectOrientation(BaseAction):
    def __init__(self, localizer=None):
        description = """
            Estimate 3D orientation of objects in an image using Orient-Anything.

            Measures:
            - Azimuth: Horizontal rotation (0-360° clockwise)
            - Polar: Vertical inclination (0-180°)
            - Rotation: In-plane rotation (-180° to +180°)
            - Confidence: Reliability score

            Useful for 3D understanding, pose estimation, and spatial reasoning.
        """

        args_spec = {"image": "Image to analyze.", "objects": "Object description(s) to analyze; string or list."}

        rets_spec = {
            "results": "List of dicts with object orientation data: [{'object': str, 'angle_data': {'azimuth': float, 'polar': float, 'rotation': float, 'confidence': float}, 'error': str or None}]"
        }

        examples = [{"name": "GetObjectOrientation", "arguments": {"image": "image-0", "objects": "a red car"}}]
        
        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        from OrientAnything.vision_tower import DINOv2_MLP
        self.dino = DINOv2_MLP(
            dino_mode='large',
            in_dim=1024,
            out_dim=360+180+180+2,  # azimuth + polar + rotation + confidence
            evaluate=True,
            mask_dino=False,
            frozen_back=False
        )
        self.dino.eval()
        self.dino.load_state_dict(torch.load('./checkpoints/Orient-Anything/croplargeEX2/dino_weight.pt', map_location='cpu'))
        self.dino = self.dino.to(self.device)
        
        self.preprocessor = AutoImageProcessor.from_pretrained("./checkpoints/dinov2-large")
        self.localize = localizer or LocalizeObjects()

    def get_object_orientation(self, image_pil, bbox):
        try:
            W, H = image_pil.size
            # Ensure bbox coordinates are within image bounds
            x1, y1, x2, y2 = [max(0, min(int(val), dim)) for val, dim in zip(bbox, [W, H, W, H])]
            if x2 <= x1 or y2 <= y1:  # Invalid bbox
                return None
                
            # Crop and process image
            cropped_image = image_pil.crop((x1, y1, x2, y2))
            
            # Ensure minimum size for Orient-Anything
            min_size = 224
            crop_w, crop_h = cropped_image.size
            if crop_w < min_size or crop_h < min_size:
                scale = max(min_size / crop_w, min_size / crop_h)
                new_w, new_h = int(crop_w * scale), int(crop_h * scale)
                cropped_image = cropped_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
            # Get angles
            from OrientAnything.inference import get_3angle
            angles = get_3angle(cropped_image, self.dino, self.preprocessor, self.device)
            
            # Process angles data
            azimuth, polar, rotation, confidence = float(angles[0]), float(angles[1]), float(angles[2]), float(angles[3])

            return {"azimuth": round(azimuth, 2), "polar": round(polar, 2),  "rotation": round(rotation, 2), "confidence": round(confidence, 4)}
        
        except Exception as e:
            return None

    def __call__(self, image: str, objects) -> Dict:
        # Normalize input to always be a list
        if isinstance(objects, str):
            objects = [objects]
            
        # Load image once for efficiency
        image_pil = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
        
        # Get object regions using localizer
        localize_results = self.localize(image=image, objects=objects)
        
        # Process each object
        results = []
        for obj in objects:
            result = {"object": obj, "angle_data": None, "error": None}
            
            # Find matching regions for this object
            matching_regions = [r for r in localize_results["regions"] if r["label"] == obj]
            
            if not matching_regions:
                result["error"] = "Object not detected"
                results.append(result)
                continue
                
            for region in matching_regions:
                bbox = region["bbox"]
            
                # Get orientation for this object
                angle_data = self.get_object_orientation(image_pil, bbox)
                
                if angle_data is None:
                    result["error"] = "Orientation estimation failed"
                else:
                    result["angle_data"] = angle_data
                    
                results.append(result)
            
        torch.cuda.empty_cache()
        
        return {"results": results}


class Get3DDistance(BaseAction):
    def __init__(self):
        description = """
            Calculates the absolute 3D spatial distance (in meters) between two pixel points (x, y) in an image. 
            Note:this tool should be used in outdoor scenes.
            Returns the calculated distance (in meters).
        """
        args_spec = {
            "image": "Path to the input image.",
            "point_1": "List of [x, y] pixel coordinates for the first point.",
            "point_2": "List of [x, y] pixel coordinates for the second point."
        }
        rets_spec = {"distance_meters": "The calculated 3D distance (float, in meters)."}
        examples = [
            {"name": "Get3DDistance", "arguments": {"image": "image-0", "point_1": [100, 100], "point_2": [1000, 1000]}},
        ]
        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)
        
        try:
            from map_anything.scripts.tool_for_spatial import MapAnythingEstimator
            self.estimator = MapAnythingEstimator()
            print("[Get3DDistance] MapAnythingEstimator initialized successfully")
        except Exception as e:
            print(f"[Get3DDistance] Warning: Failed to initialize MapAnythingEstimator: {e}")
            self.estimator = None

    def __call__(self, image, point_1, point_2, visualize_output=False):
        print(f"[Get3DDistance] Processing image: {image}, point1: {point_1}, point2: {point_2}")
        distance = None
        abs_visualization_path = None
        error_message = None

        try:
            # 1. Validate input point coordinates are integers
            if not (isinstance(point_1, list) and len(point_1) == 2 and all(isinstance(i, (int, float)) for i in point_1)) or \
               not (isinstance(point_2, list) and len(point_2) == 2 and all(isinstance(i, (int, float)) for i in point_2)):
                raise ValueError("Point coordinates must be a list containing two integers [x, y]")

            # 2. Get original image dimensions (part of the tool's logic)
            try:
                with Image.open(image) as img:
                    original_shape_WH = img.size # (width, height)
                print(f"[Get3DDistance] Original dimensions (W, H): {original_shape_WH}")
            except Exception as e:
                raise IOError(f"Cannot read image dimensions: {e}") # Re-raise as IOError

            # 3. Convert coordinates to tuples (x, y)
            p1_coords, p2_coords = tuple(point_1), tuple(point_2)

            # 4. Call the engine to calculate distance
            distance = self.estimator.get_distance_between_points(image, p1_coords, p2_coords, original_shape_WH)
            print(f"[Get3DDistance] Calculated distance: {distance:.4f} meters")
            
            if visualize_output:
                # 5. Save visualization image
                output_dir = "./output" # You can change the output directory if needed
                os.makedirs(output_dir, exist_ok=True)
                base_filename = os.path.splitext(os.path.basename(image))[0]
                # More detailed filename including coordinates
                vis_filename = f"{base_filename}_distance_vis_p{point_1[0]}-{point_1[1]}_p{point_2[0]}-{point_2[1]}.png"
                visualization_path = os.path.join(output_dir, vis_filename)

                try:
                    img_cv = cv2.imread(image)
                    if img_cv is None:
                        raise IOError(f"cv2 failed to read image: {image}")

                    color = (0, 0, 255) # BGR Red
                    thickness = 2
                    radius = 5 # Radius of the drawn points

                    # Draw points and line (ensure coordinates are integers)
                    cv2.circle(img_cv, (int(p1_coords[0]), int(p1_coords[1])), radius, color, -1) # -1 for filled circle
                    cv2.circle(img_cv, (int(p2_coords[0]), int(p2_coords[1])), radius, color, -1)
                    cv2.line(img_cv, (int(p1_coords[0]), int(p1_coords[1])), (int(p2_coords[0]), int(p2_coords[1])), color, thickness)

                    cv2.imwrite(visualization_path, img_cv)

                except Exception as e:
                    print(f"[Get3DDistance] Failed to save point visualization image: {e}")

        except (ValueError, IOError, Exception) as e:
             print(f"[Get3DDistance] Error during processing: {e}")
             error_message = str(e)

        return {"distance_meters": distance, "error": error_message}


class Calculate(BaseAction):
    def __init__(self) -> None:
        description = "Calculate a mathematical expression."
        args_spec = {"expression": "The mathematical expression to calculate (e.g., '2 + 2', '10 * 5.5')."}
        rets_spec = {"result": "The numerical result of the calculation."}
        examples = [
            {"name": "Calculate", "arguments": {"expression": "2 + 2"}}, 
            {"name": "Calculate", "arguments": {"expression": "4 * 9 * 84"}},
            {"name": "Calculate", "arguments": {"expression": "5 - 4 / 2"}},
        ]

        super().__init__(description=description, args_spec=args_spec, rets_spec=rets_spec, examples=examples)
    
    def __call__(self, expression):
        result = eval(expression)

        return {"result": result}

def main():
    # CUDA_VISIBLE_DEVICES=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TORCH_HUB_OFFLINE=1 python action.py
    import time
    img1 = "../SpatialScore_benchmark/Q-Spatial-Bench/QSpatial_plus/images/018.jpeg"
    img2 = "../SpatialScore_benchmark/3DSRBench/coco_images/train2017/000000002400.jpg"
    img3 = "../SpatialScore_benchmark/3DSRBench/coco_images/train2017/000000002400.jpg"
            
    examples = [
        {"name": "Terminate", "arguments": {"answer": "This is my final answer"}},
        # {"name": "Calculate", "arguments": {"expression": "3 * 4 + 10 / 2"}},
        # {"name": "MatchImagesSIFT", "arguments": {"image": [img1, img2], "num_keypoints": 500, "ratio_th": 0.75}},
        # {"name": "EstimateHomographyMatrix", "arguments": {"image": [img1, img1], "num_keypoints": 500, "ratio_th": 0.75}},
        # {"name": "CountObjects", "arguments": {"image": img2, "objects": ["dog", "cat"]}},
        # {"name": "LocalizeObjects", "arguments": {"image": img2, "objects":["dog", "cat"]}, "visualize_output": False},
        # {"name": "EstimateOpticalFlow", "arguments": {"image": [img2, img3]}},
        # {"name": "SelfThinking", "arguments": {"query": "What is the capital of the USA?"}},
        # {"name": "SelfThinking", "arguments": {"query": "How many sidewalks are in the image?", "image": [img1]}},
        # {"name": "EstimateImageDepth", "arguments": {"image": img1, "indoor_or_outdoor": "outdoor"}},
        # {"name": "EstimateRegionDepth", "arguments": {"image": img1, "bboxes": [125, 125, 300, 300], "indoor_or_outdoor": "outdoor"}},
        # {"name": "EstimateObjectDepth", "arguments": {"image": img1, "objects": ["bus", "truck"], "indoor_or_outdoor": "outdoor"}},
        # {"name": "GetObjectOrientation", "arguments": {"image": img2, "objects": "dog"}},
        # {"name": "GetCameraParametersVGGT", "arguments": {"image": [img2, img3], "dtype": "float16"}},
        # {"name": "EstimateObjectGeometryProperties", "arguments": {"image": img2, "object_descs": ["dog", "person"]}},
        # {"name": "Get3DDistance", "arguments": {"image": img1, "point_1": [1483.243, 1444.905], "point_2": [2138.594, 1521.594],"visualize_output": True}},        
        # {"name": "GetObjectMask", "arguments": {"image": img2, "objects": ["dog", "cat"]}},
        # {"name": "Detect3DObjects", "arguments": {"image": img3, "objects": ["person"], "visualize_output": False}},
    ]

    # for i, function in enumerate(examples):
    #     function_args = function['arguments']
    #     action = globals()[function['name']]()
    #     results = action(**function_args)
    #     print(results)

    for i, function in enumerate(examples):
        function_name = function['name']
        function_args = function['arguments']
        
        print(f"\n{'='*60}")
        print(f"Testing: {function_name}")
        print(f"{'='*60}")
        
        action = globals()[function_name]()
        
        _ = action(**function_args)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        num_iterations = 100
        times = []
        
        for iteration in range(num_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            results = action(**function_args)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if (iteration + 1) % 10 == 0:
                print(f"Progress: {iteration + 1}/{num_iterations}")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_used = peak_memory - initial_memory
        else:
            peak_memory = 0
            current_memory = 0
            memory_used = 0
        
        print(f"\n{'─'*60}")
        print(f"Results for {function_name}:")
        print(f"  Average Time: {avg_time*1000:.2f} ms")
        print(f"  Min Time:     {min_time*1000:.2f} ms")
        print(f"  Max Time:     {max_time*1000:.2f} ms")
        print(f"  Peak Memory:  {peak_memory:.2f} MB")
        print(f"  Current Mem:  {current_memory:.2f} MB")
        print(f"  Memory Used:  {memory_used:.2f} MB")
        print(f"{'─'*60}\n")
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


