import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from utils.util import image_to_base64_data_uri, extract_number, extract_yes_no, extract_option, extract_numeric_with_unit, load_image

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Optimize GPU memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SEED = 42
torch.manual_seed(SEED)

# Define the root path for visual data
DATA_ROOT = "./SpatialScore_released/SpatialScore_benchmark"

def load_model_and_components(model_path, model_name="qwen2_5vl-7b"):
    if model_name in ["qwen2_5vl-3b", "qwen2_5vl-7b", "qwen2_5vl-32b", "qwen2_5vl-72b", "spaceqwen-3b", "spacethinker-qwen2_5vl-3b", "SpaceR"]:
        # Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-72B-Instruct, SpaceQwen-3B-Instruct
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True, min_pixels=256*28*28, max_pixels=2560*28*28)
    elif model_name in ["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "qwen3vl-32b"]:
        model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2", trust_remote_code=True)
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True, min_pixels=256*28*28, max_pixels=2560*28*28)

    return model, tokenizer_or_processor

def process_model_input(sample):
    question_type = sample.get('question_type', '').lower()
    
    # Define prompts based on question type
    if question_type == 'multi-choice':
        assistant_prompt = (
            "**Please select the most appropriate answer from the given options.**\n"
            "**Respond ONLY with the capital letter and its parentheses.**\n\nQuestion: "
        )
    elif question_type == 'judgement':
        assistant_prompt = (
            "**Please answer with yes or no based on the image.**\n"
            "**Respond ONLY with 'yes' or 'no'.**\n\nQuestion: "
        )
    elif question_type == 'open-ended':
        # Check if it's a distance measurement question
        extra_info = sample.get('extra_info', {})
        has_measurement_fields = 'answer_value' in extra_info and 'answer_unit' in extra_info
        
        if has_measurement_fields:
            assistant_prompt = (
                "Please answer the question by measuring the precise distance in 3D space through 2D images or videos.\n"
                "Respond ONLY with a numeric answer consisting of a scalar and a distance unit in the format of **\\scalar{scalar} \\distance_unit{distance unit}**.\n\nQuestion: "
            )
        else:
            assistant_prompt = (
                "**Please answer the question based on the given image or video.**\n"
                "**Respond ONLY with a concise and accurate scalar or a scalar with corresponding unit.**\n\nQuestion: "
            )
    
    # Process images - support multiple images
    img_paths = sample.get('img_paths', sample.get('image_paths', []))
    if isinstance(img_paths, str):
        img_paths = [img_paths]
    
    processed_paths = []
    for path in img_paths:
        if os.path.isabs(path):
            abs_path = path
        else:
            # Join with DATA_ROOT, handling the "./" prefix if present
            clean_rel_path = path[2:] if path.startswith("./") else path
            abs_path = os.path.join(DATA_ROOT, clean_rel_path)
        processed_paths.append(abs_path)
    
    images = [Image.open(p).convert('RGB') for p in processed_paths]
    sample['images'] = images
    sample['img_paths'] = processed_paths # Store the absolute paths
    sample['assistant_prompt'] = assistant_prompt

    return sample

def load_and_process_data(dataset_json_path):
    """Load NDJSON data and process multi-choice questions"""
    data = []
    with open(dataset_json_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                item = json.loads(line.strip())
                
                # Process multi-choice questions
                if item.get('question_type') == 'multi-choice' and 'options' in item:
                    # Combine question with formatted options
                    question = item.get('question', '')
                    options = item.get('options', [])
                    
                    # Format options as (A), (B), (C), etc.
                    option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
                    formatted_options = []
                    for i, option in enumerate(options):
                        if i < len(option_letters):
                            formatted_options.append(f"({option_letters[i]}) {option}")
                    
                    # Combine question with options
                    if formatted_options:
                        combined_question = f"{question}\n" + "\n".join(formatted_options)
                        item['question'] = combined_question
                
                # Ensure img_paths is available and is a list
                if 'image_paths' in item:
                    item['img_paths'] = item['image_paths']
                elif 'img_paths' not in item and 'image_path' in item:
                    item['img_paths'] = [item['image_path']]
                
                # Use source_dataset field for consistency
                if 'source_dataset' in item and 'source' not in item:
                    item['source'] = item['source_dataset']
                
                data.append(item)
    
    return data

def generate_response(model, tokenizer_or_processor, model_inputs):
    NUM_BEAMS, TEMPERATURE, MAX_NEW_TOKENS, USE_CACHE = 1, 0.0, 512, True
    DO_SAMPLE = True if TEMPERATURE > 0 else False
    
    device = next(model.parameters()).device
    
    assistant_prompt, image_paths, images, text = model_inputs['assistant_prompt'], model_inputs['img_paths'], model_inputs['images'], model_inputs['question']

    with torch.no_grad():
        image_content = [{"type": "image", "image": img_path} for img_path in image_paths]

        messages = [
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": image_content + [{"type": "text", "text": text}]}
        ]
        
        input_text = tokenizer_or_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = tokenizer_or_processor(text=[input_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

        output_ids = model.generate(**inputs, num_beams=NUM_BEAMS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, use_cache=USE_CACHE, do_sample=DO_SAMPLE)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        response = tokenizer_or_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
    torch.cuda.empty_cache()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="SpatialScore Evaluation")
    parser.add_argument('--model_path', type=str, default="../huggingface/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--model_name', type=str, default='qwen2_5vl-7b', choices=['qwen2_5vl-3b', 'qwen2_5vl-7b', 'qwen2_5vl-32b', "qwen2_5vl-72b", "qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "qwen3vl-32b", "spaceqwen-3b", "spacethinker-qwen2_5vl-3b", "SpaceR"])
    parser.add_argument('--dataset_json_path', type=str, default="./SpatialScore_benchmark/SpatialScore_benchmark.ndjson")
    parser.add_argument('--output_dir', type=str, default="./eval_results")
    parser.add_argument('--save_interval', type=int, default=50)
    
    args = parser.parse_args()

    model, tokenizer_or_processor = load_model_and_components(args.model_path, args.model_name)

    # Load and process NDJSON data
    data = load_and_process_data(args.dataset_json_path)
    
    print(f"Loaded {len(data)} items from unified benchmark")

    # Create output directory structure
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    all_results = []
    total_samples = 0

    # Check for existing results to resume from
    start_idx = 0
    results_file = os.path.join(output_dir, "all_results.json")
    
    # Resume from existing results if requested
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        
        # Determine starting point for processing
        processed_ids = set(result.get('id') for result in all_results if result.get('id') is not None)
        if processed_ids:
            # Find items that haven't been processed yet
            unprocessed_items = [item for item in data if item.get('id') not in processed_ids]
            if not unprocessed_items:
                print("All items have been processed already.")
                return
            start_idx = len(data) - len(unprocessed_items)
        
        print(f"Resuming from index {start_idx}, found {len(all_results)} existing results")

    # Process items starting from start_idx
    for i, item in enumerate(tqdm(data[start_idx:], desc="Processing SpatialScore items", initial=start_idx, total=len(data))):
        actual_idx = i + start_idx  # Calculate the actual index in the full dataset
        
        # Skip if this item has already been processed (as a safeguard)
        if any(r.get('id') == item.get('id') for r in all_results):
            continue

        # Process model input and generate response
        model_inputs = process_model_input(item)
        response = generate_response(model, tokenizer_or_processor, model_inputs)
        
        # Create result entry without evaluation
        result_entry = {
            "id": item.get('id', actual_idx),
            "source_dataset": item.get('source_dataset', item.get('source', 'unknown')),
            "category": item.get('category', 'unknown'), 
            "task": item.get('task', 'unknown'),
            "sub_task": item.get('sub_task', 'unknown'),
            "input_modality": item.get('input_modality', 'single-image'), 
            "question_type": item.get('question_type', 'unknown'),
            "question": item.get('question', ''),
            "options": item.get('options', []),
            "extra_info": item.get('extra_info', {}),
            "img_paths": item.get('img_paths', item.get('image_paths', [])),
            "video_path": item.get('video_path', None),
            "gt_answer": item.get('answer', ''),
            "pred_answer": response,
        }
        
        # Add to all results
        all_results.append(result_entry)
        total_samples += 1
        
        # Save intermediate results periodically
        if (i + 1) % args.save_interval == 0:
            # Save all results so far
            with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"Saved intermediate results: {len(all_results)} items processed")

        torch.cuda.empty_cache()
        
    # Save final results
    with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save a simple summary
    summary = {
        "model_name": args.model_name,
        "dataset_path": args.dataset_json_path,
        "total_processed": len(all_results),
        "total_items": len(data)
    }
    
    with open(os.path.join(output_dir, "processing_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Processing completed: {len(all_results)} items processed out of {len(data)} total items")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()