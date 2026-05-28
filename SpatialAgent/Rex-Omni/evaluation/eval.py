import argparse
import json
import os
import re

from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

DEFAULT_PROMPT = "You are a helpful assistant"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="work_dirs/det2.0/_1_coco_only_baseline/_1.7_new_special_token/special_token",
    )
    parser.add_argument(
        "--test_jsonl_path",
        type=str,
        default="mounted_files/det2.0/benchmark/merged_common_longtail_referring_vp_500subset.jsonl",
        help="Path to the test JSONL file containing benchmark data",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        default="",
        help="Root directory to prepend to image paths. If empty, use image_path as is.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="work_dirs/det2.0/_1_coco_only_baseline/_1.7_new_special_token/special_token/eval_results/benchmark_eval.jsonl",
    )
    parser.add_argument(
        "--output_box_format",
        type=str,
        default="xyxy",
        choices=["xyxy", "cxcywh", "xywh"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=2560 * 28 * 28)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument(
        "--end_idx",
        type=int,
        default=-1,
        help="End index for processing, -1 means process all",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
        choices=[
            "Detect [OBJ]. Output the bounding box coordinates in [x0, y0, x1, y1] format.",  # for box prompt detection
            "point to [OBJ].",  # for point prompt detection
            "Can you detect all the [OBJ] in this image in box format like [x0, y0, x1, y1] and then recognize them?",  # for ocr in box format
            "Can you detect all the [OBJ] in this image in polygon format like [x0, y0, x1, y1, x2, y2 ...] and then recognize them?",  # for ocr in polygon format
            "Given reference boxes [OBJ] indicating one or more objects, find all objects with the same category in the image and output their bounding boxes in [x0, y0, x1, y1] format.",  # visual prompt evaluation
        ],
        help="Question template with [OBJ] placeholder for prompting",
    )
    parser.add_argument(
        "--system_prompt", type=str, default="You are a helpful assistant"
    )
    return parser.parse_args()


def inference(
    image,
    prompt,
    system_prompt="DEFAULT_PROMPT",
    max_new_tokens=2048,
    min_pixels=16 * 28 * 28,
    max_pixels=1280 * 28 * 28,
):
    system_prompt = DEFAULT_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to("cuda")

    image_inputs, video_inputs = process_vision_info(messages)

    mm_data = {}
    mm_data["image"] = image_inputs
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = model.generate([llm_inputs], sampling_params=sampling_params)

    generated_text = outputs[0].outputs[0].text

    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14

    return generated_text, input_height, input_width, text


def parse_prediction(
    text, w, h, input_height, input_width, output_box_format, task_name=None
):
    """
    Parse model output text to extract category-wise bounding boxes or keypoints.
    Input format example:
    "<|object_ref_start|>cell phone<|object_ref_end|><|box_start|>[[298, 143, 365, 279]]<|box_end|>, <|object_ref_start|>person<|object_ref_end|><|box_start|>[[1, 10, 419, 636], [263, 18, 419, 298]]<|box_end|><|im_end|>"

    For keypoint detection, expects JSON format with bbox and keypoints.

    Returns:
    {
        'category1': [[x1,y1,x2,y2], [x1,y1,x2,y2], ...],  # for bounding boxes
        'category2': [x1,y1,x2,y2,x3,y3,...],  # for polygons/points
        ...
    }
    or for keypoint detection:
    {
        'category1': [{'bbox': [...], 'keypoints': {...}, 'phrase': '...'}],
        ...
    }
    """
    result = {}
    # Handle keypoint detection tasks
    if task_name == "keypoint":
        return parse_keypoint_prediction(text, w, h)

    # Remove the end marker if present
    text = text.split("<|im_end|>")[0]

    # First, try the standard pattern with complete box_start/box_end pairs
    pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>"
    matches = re.findall(pattern, text)

    # Process standard matches
    for category, box_text in matches:
        result = _process_box_text(category, box_text, result, w, h)

    # Handle special case: incomplete box_end for single object_ref_start
    # Check if there's exactly one object_ref_start and it might be missing box_end
    object_ref_count = text.count("<|object_ref_start|>")

    if object_ref_count == 1 and not matches:
        # Try to extract incomplete pattern
        incomplete_pattern = r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>(.*?)$"
        incomplete_match = re.search(incomplete_pattern, text, re.DOTALL)

        if incomplete_match:
            category, box_text = incomplete_match.groups()
            print(
                f"Found incomplete prediction for category '{category.strip()}', attempting to extract coordinates..."
            )
            result = _process_box_text(
                category, box_text, result, w, h, is_incomplete=True
            )

    return result


def _process_box_text(category, box_text, result, w, h, is_incomplete=False):
    """
    Helper function to process box text and extract coordinates
    """
    category = category.strip()

    if is_incomplete:
        # For incomplete predictions, be more lenient with parsing
        # Remove any trailing incomplete content that might cause issues
        box_text = box_text.strip()
        # Remove trailing commas or incomplete coordinates
        if box_text.endswith(","):
            box_text = box_text[:-1]

    # 按逗号分割多个polygon序列
    polygon_sequences = [seq.strip() for seq in box_text.split(",") if seq.strip()]

    boxes = []
    for polygon_seq in polygon_sequences:
        # Find all box tokens in the format <{number}>
        box_pattern = r"<(\d+)>"
        box_matches = re.findall(box_pattern, polygon_seq)

        if not box_matches:
            continue

        # Process coordinates flexibly - handle different numbers of coordinates
        i = 0
        while i < len(box_matches):
            try:
                # Try to determine the number of coordinates for this shape
                # For bounding boxes: 4 coordinates (x0, y0, x1, y1)
                # For points: 2 coordinates (x, y)
                # For polygons: multiple pairs of coordinates

                # First, try to find a complete shape by looking ahead
                coords = []
                j = i

                # For bounding boxes (4 coordinates)
                if j + 3 < len(box_matches):
                    # Check if this looks like a bounding box (4 consecutive coordinates)
                    x0_bin = int(box_matches[j])
                    y0_bin = int(box_matches[j + 1])
                    x1_bin = int(box_matches[j + 2])
                    y1_bin = int(box_matches[j + 3])

                    # Validate that coordinates make sense (x1 > x0, y1 > y0)
                    if x1_bin > x0_bin and y1_bin > y0_bin:
                        # Convert from bins [0, 999] back to normalized coordinates [0, 1]
                        x0_norm = x0_bin / 999.0
                        y0_norm = y0_bin / 999.0
                        x1_norm = x1_bin / 999.0
                        y1_norm = y1_bin / 999.0

                        # Convert normalized coordinates directly to target image size
                        x0 = x0_norm * w
                        y0 = y0_norm * h
                        x1 = x1_norm * w
                        y1 = y1_norm * h

                        coords = [x0, y0, x1, y1]
                        i += 4
                    else:
                        # If not a valid bounding box, try as individual coordinates
                        coords = []
                        while j < len(box_matches):
                            try:
                                coord_bin = int(box_matches[j])
                                coord_norm = coord_bin / 999.0
                                if j % 2 == 0:  # x coordinate
                                    coord = coord_norm * w
                                else:  # y coordinate
                                    coord = coord_norm * h
                                coords.append(coord)
                                j += 1
                            except (ValueError, IndexError):
                                break
                        i = j
                else:
                    # Handle remaining coordinates as individual points or polygon vertices
                    # For incomplete predictions, try to extract partial coordinates
                    coords = []
                    remaining_coords = len(box_matches) - j

                    if is_incomplete and remaining_coords >= 2:
                        # For incomplete predictions, if we have at least 2 coordinates, try to use them
                        # But ensure we have an even number for valid (x,y) pairs
                        valid_coords_count = (remaining_coords // 2) * 2

                        for k in range(valid_coords_count):
                            try:
                                coord_bin = int(box_matches[j + k])
                                coord_norm = coord_bin / 999.0
                                if k % 2 == 0:  # x coordinate
                                    coord = coord_norm * w
                                else:  # y coordinate
                                    coord = coord_norm * h
                                coords.append(coord)
                            except (ValueError, IndexError):
                                break
                        i = j + valid_coords_count

                        if is_incomplete and len(coords) == 2:
                            # If we only have 2 coordinates, treat as a point
                            pass
                        elif (
                            is_incomplete and len(coords) >= 4 and len(coords) % 2 == 0
                        ):
                            # If we have 4 or more even coordinates, it could be a valid shape
                            pass
                        else:
                            # Skip invalid coordinate sets
                            coords = []
                    else:
                        # Standard processing for complete predictions
                        while j < len(box_matches):
                            try:
                                coord_bin = int(box_matches[j])
                                coord_norm = coord_bin / 999.0
                                if j % 2 == 0:  # x coordinate
                                    coord = coord_norm * w
                                else:  # y coordinate
                                    coord = coord_norm * h
                                coords.append(coord)
                                j += 1
                            except (ValueError, IndexError):
                                break
                        i = j

                # Add the coordinates if we found any
                if coords:
                    if is_incomplete:
                        print(
                            f"Extracted {len(coords)} coordinates for incomplete prediction: {coords[:8]}..."
                        )  # Show first 8 coords
                    boxes.append(coords)

            except (ValueError, IndexError) as e:
                if is_incomplete:
                    print(f"Error parsing incomplete box coordinates: {e}")
                else:
                    print(f"Error parsing box coordinates: {e}")
                i += 1
                continue

    # Add boxes for this category
    if category not in result:
        result[category] = []
    result[category].extend(boxes)

    return result


def parse_keypoint_prediction(text, w, h):
    """
    Parse keypoint task JSON output to extract bbox and keypoints.

    Expected format:
    ```json
    {
        "person1": {
            "bbox": " <1> <36> <987> <984> ",
            "keypoints": {
                "nose": " <540> <351> ",
                "left eye": " <559> <316> ",
                "right eye": "unvisible",
                ...
            }
        },
        ...
    }
    ```

    Returns:
    Dict with category as key and list of keypoint instances as value
    """
    import json
    import re

    # Extract JSON content from markdown code blocks
    json_pattern = r"```json\s*(.*?)\s*```"
    json_matches = re.findall(json_pattern, text, re.DOTALL)

    if not json_matches:
        # Try to find JSON without markdown
        try:
            # Look for JSON-like structure
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
            else:
                return {}
        except:
            return {}
    else:
        json_str = json_matches[0]

    try:
        keypoint_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing keypoint JSON: {e}")
        return {}

    result = {}

    for instance_id, instance_data in keypoint_data.items():
        if "bbox" not in instance_data or "keypoints" not in instance_data:
            continue

        bbox = instance_data["bbox"]
        keypoints = instance_data["keypoints"]

        # Convert bbox coordinates from bins [0, 999] to absolute coordinates
        if isinstance(bbox, str) and bbox.strip():
            # Parse box tokens from string format like " <1> <36> <987> <984> "
            coord_pattern = r"<(\d+)>"
            coord_matches = re.findall(coord_pattern, bbox)

            if len(coord_matches) == 4:
                try:
                    x0_bin, y0_bin, x1_bin, y1_bin = [
                        int(match) for match in coord_matches
                    ]
                    x0 = (x0_bin / 999.0) * w
                    y0 = (y0_bin / 999.0) * h
                    x1 = (x1_bin / 999.0) * w
                    y1 = (y1_bin / 999.0) * h
                    converted_bbox = [x0, y0, x1, y1]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing bbox coordinates: {e}")
                    continue
            else:
                print(
                    f"Invalid bbox format for {instance_id}: expected 4 coordinates, got {len(coord_matches)}"
                )
                continue
        else:
            print(f"Invalid bbox format for {instance_id}: {bbox}")
            continue

        # Convert keypoint coordinates from bins to absolute coordinates
        converted_keypoints = {}
        for kp_name, kp_coords in keypoints.items():
            if kp_coords == "unvisible" or kp_coords is None:
                converted_keypoints[kp_name] = "unvisible"
            elif isinstance(kp_coords, str) and kp_coords.strip():
                # Parse box tokens from string format like " <540> <351> "
                coord_pattern = r"<(\d+)>"
                coord_matches = re.findall(coord_pattern, kp_coords)

                if len(coord_matches) == 2:
                    try:
                        x_bin, y_bin = [int(match) for match in coord_matches]
                        x = (x_bin / 999.0) * w
                        y = (y_bin / 999.0) * h
                        converted_keypoints[kp_name] = [x, y]
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing keypoint coordinates for {kp_name}: {e}")
                        converted_keypoints[kp_name] = "unvisible"
                else:
                    print(
                        f"Invalid keypoint format for {kp_name}: expected 2 coordinates, got {len(coord_matches)}"
                    )
                    converted_keypoints[kp_name] = "unvisible"
            else:
                converted_keypoints[kp_name] = "unvisible"

        # Group by category (assuming instance_id contains category info)
        # Try to extract category from instance_id (e.g., "person1" -> "person")
        category = "keypoint_instance"
        if instance_id:
            # Remove numbers from instance_id to get category
            import re

            category_match = re.match(r"^([a-zA-Z_]+)", instance_id)
            if category_match:
                category = category_match.group(1)

        if category not in result:
            result[category] = []

        result[category].append(
            {
                "bbox": converted_bbox,
                "keypoints": converted_keypoints,
                "phrase": instance_id,  # Use instance_id as phrase
            }
        )

    return result


def load_test_data(test_jsonl_path):
    """
    Load test data from JSONL file.

    Returns:
    List of test entries, each containing:
    {
        'image_name': str,
        'question': str,
        'gt': dict,
        'categories': list,
        'task_name': str,
        'dataset_name': str,
        'visual_prompt': dict (optional, for visual prompt tasks)
    }
    """
    test_data = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    test_data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")

    return test_data


if __name__ == "__main__":
    args = get_args()
    if not (args.start_idx == -1 and args.end_idx == -1):
        args.save_path = args.save_path.replace(
            ".jsonl",
            f"_{args.start_idx}_{args.end_idx}.jsonl",
        )

    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    model = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tokenizer_mode="slow",
        limit_mm_per_prompt={"image": 10, "video": 10},
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    # Define stop sequences to prevent repetitive coordinate predictions
    stop_sequences = [
        "<|im_end|>",  # Standard end token
        # Add patterns for repetitive coordinates
        "<0><0><0><0>",  # Repeated zero coordinates
        # Add more patterns based on your observation
    ]

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=0.8,
        repetition_penalty=1.05,  # Increased to reduce repetition
        top_k=1,
        temperature=0.0,
        skip_special_tokens=False,
        stop=stop_sequences,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )

    # Load test data from JSONL file
    print(f"Loading test data from: {args.test_jsonl_path}")
    test_data = load_test_data(args.test_jsonl_path)
    print(f"Loaded {len(test_data)} test entries")

    # Apply start/end index filtering
    if args.end_idx == -1:
        selected_test_data = test_data[args.start_idx :]
    else:
        selected_test_data = test_data[args.start_idx : args.end_idx]

    print(
        f"Processing {len(selected_test_data)} entries (from index {args.start_idx} to {args.end_idx if args.end_idx != -1 else 'end'})"
    )

    predictions = []

    for entry in tqdm(selected_test_data, desc="Processing test entries"):
        image_path = entry["image_path"]
        categories = entry["categories"]
        gt = entry["gt"]
        dataset_name = entry["dataset_name"]

        # Construct full image path by joining image_root_dir with image_path
        if args.image_root_dir:
            full_image_path = os.path.join(args.image_root_dir, image_path)
        else:
            full_image_path = image_path

        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}")
            continue

        try:
            image = Image.open(full_image_path).convert("RGB")
            w, h = image.size
        except Exception as e:
            print(f"Error loading image {full_image_path}: {e}")
            continue

        # Replace [OBJ] placeholder with categories
        question = args.question.replace("[OBJ]", ", ".join(categories))

        output, input_height, input_width, text = inference(
            image,
            question,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

        print(f"[{dataset_name}]")
        print(f"Question: {question}")
        print(f"Output: {output}")

        try:
            extracted_predictions = parse_prediction(
                output,
                w,
                h,
                input_height.item(),
                input_width.item(),
                args.output_box_format,
            )

            prediction = {
                "image_path": image_path,
                "extracted_predictions": extracted_predictions,
                "gt": gt,
                "question": question,
                "dataset_name": dataset_name,
                "raw_response": output,
            }

        except Exception as e:
            print(f"Parse failed, error is {e}")
            prediction = {
                "image_path": image_path,
                "extracted_predictions": {},
                "gt": gt,
                "question": question,
                "dataset_name": dataset_name,
                "raw_response": output,
            }

        predictions.append(prediction)

    # Save predictions
    print(f"\n💾 Saving predictions to: {args.save_path}")
    with open(args.save_path, "a") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

    print(f"✅ Saved {len(predictions)} predictions!")
