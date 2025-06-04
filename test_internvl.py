import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils.util import image_to_base64_data_uri, extract_number, extract_yes_no, extract_option, extract_numeric_with_unit, load_image

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Optimize GPU memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SEED = 42
torch.manual_seed(SEED)

def load_model_and_components(model_path, model_name="internvl3-8b"):
    if model_name in ["internvl2_5-1b", "internvl2_5-4b", "internvl2_5-8b", "internvl2_5-38b", "internvl2_5-78b", "internvl3-1b", "internvl3-8b", "internvl3-14b", "internvl3-38b", "internvl3-78b"]:
        # InternVL2_5-1B, InternVL2_5-4B, InternVL2_5-8B, InternVL3-1B, InternVL3-8B, InternVL3-14B, InternVL3-38B, InternVL3-78B
        model = AutoModel.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    return model, tokenizer_or_processor

def process_model_input(sample):
    dataset_name = sample.get('source', 'unknown')
    # Define common system prompt for multiple choice questions
    if dataset_name in ["cvbench", "MMIU", "BLINK", "3DSRBench", "MMVP"]:
        assistant_prompt = (
            "**Please select the most appropriate answer from options (A), (B), (C), (D), (E), or (F).**\n"
            "**Respond ONLY with the letter and its parentheses, for example: (A)**\n\nQuestion: "
        )
    elif dataset_name in ["spatialbench", "VSR-ZeroShot", "VSR-Random", "SpatialSense", "VSI-Bench_8", "RealWorldQA"]:
        assistant_prompt = ("**Answer concisely with a single word, number, or option (e.g., yes, no, 5, 2.2, A).**\n\nQuestion: ")
    elif dataset_name in ["QSpatialBench-Plus", "QSpatialBench-ScanNet"]:
        assistant_prompt = (
            "You will be provided with a question and a 2D image. The question involves measuring the precise distance in 3D space through a 2D image. You will answer the question by providing a numeric answer consisting of a scalar and a distance unit in the format of **\scalar{scalar} \distance_unit{distance unit}** at the end of your response.\n"
            "Let's think step by step and start by finding good reference objects or object parts in the image.\n\n"
            "Question:"
        )
    elif dataset_name == "VGBench":
        if sample.get('question_type') == 'open-ended':
            assistant_prompt = (
            "You will be provided with a question and a 2D image. The question involves measuring the precise distance in 3D space through a 2D image. You will answer the question by providing a numeric answer consisting of a scalar and a distance unit in the format of **\scalar{scalar} \distance_unit{distance unit}** at the end of your response.\n"
            "Let's think step by step and start by finding good reference objects or object parts in the image.\n\n"
            "Question:"
        )
        else:
            assistant_prompt = ("**Answer concisely with a single word, number, or option (e.g., yes, no, 5, 2.2, A).**\n\nQuestion: ")
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    
    # Process images - support multiple images
    images = [load_image(img_path, max_num=32) for img_path in sample['img_paths']] # max_num=128 is the maximum patches for InternVL
    sample['images'] = images
    sample['assistant_prompt'] = assistant_prompt

    return sample

def generate_response(model, tokenizer_or_processor, model_inputs):
    NUM_BEAMS, TEMPERATURE, MAX_NEW_TOKENS, USE_CACHE = 1, 0.0, 1024, True
    DO_SAMPLE = True if TEMPERATURE > 0 else False
    device = next(model.parameters()).device

    assistant_prompt, image_paths, images, text = model_inputs['assistant_prompt'], model_inputs['img_paths'], model_inputs['images'], model_inputs['question']

    with torch.no_grad():
        if len(images) == 1:
            input_text = f"<image>\n{assistant_prompt} {text}"
        else:
            input_text = "\n".join([f"Image-{i+1}: <image>" for i in range(len(images))]) + f"\n{assistant_prompt} {text}"

        num_patches_list = [image.size(0) for image in images] # Input multi-image separately
        images = torch.cat(images, dim=0).to(device)  # Concatenate images along batch dimension
        
        response = model.chat(tokenizer_or_processor, images, input_text, num_patches_list=num_patches_list, history=None, return_history=False,
            generation_config=dict(num_beams=NUM_BEAMS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE),
        )
        response = response.strip()

    torch.cuda.empty_cache()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="SpatialScore Evaluation")
    parser.add_argument('--model_path', type=str, default="./huggingface/InternVL3-8B")
    parser.add_argument('--model_name', type=str, default='internvl3-8b', choices=['internvl2_5-1b', 'internvl2_5-4b', 'internvl2_5-8b', 'internvl2_5-38b', 'internvl2_5-78b', 'internvl3-1b', 'internvl3-8b', 'internvl3-14b', 'internvl3-38b', 'internvl3-78b'])
    parser.add_argument('--dataset_json_path', type=str, default="./dataset/SpatialScore.json")
    parser.add_argument('--dataset_name', type=str, default="all", choices=['all', 'VGBench', 'cvbench', 'spatialbench', 'MMIU', 'BLINK', '3DSRBench', 'RealWorldQA', 'MMVP', 'QSpatialBench-Plus', 'QSpatialBench-ScanNet', 'VSR-ZeroShot', 'SpatialSense', 'VSI-Bench_8'])
    parser.add_argument('--output_dir', type=str, default="./eval_results")
    parser.add_argument('--use_extra_input', action='store_true')
    parser.add_argument('--save_interval', type=int, default=20)
    
    args = parser.parse_args()

    model, tokenizer_or_processor = load_model_and_components(args.model_path, args.model_name)

    with open(args.dataset_json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from unified benchmark")

    # Create output directory structure
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    all_results, source_results, category_results = [], {}, {}
    total_correct, total_samples = 0, 0

    # Check for existing results if resume flag is set
    start_idx = 0
    results_file = os.path.join(output_dir, "all_results.json")
    
    # Resume from existing results if requested
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        
        # Load existing counters from summary file if it exists
        summary_file = os.path.join(output_dir, "overall_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                total_correct = summary.get('correct', 0)
                total_samples = summary.get('total', 0)
        
        # Group existing results by source and category
        for result in all_results:
            source = result.get('source', 'unknown')
            if source not in source_results:
                source_results[source] = []
            source_results[source].append(result)
            
            category = result.get('category', 'unknown')
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        # Determine starting point for processing
        processed_ids = set(result.get('id') for result in all_results if isinstance(result.get('id'), int))
        if processed_ids:
            # Find the next index to process
            start_idx = max(processed_ids) + 1
        
        print(f"Resuming from index {start_idx}, found {len(all_results)} existing results")

    # Process items starting from start_idx
    for i, item in enumerate(tqdm(data[start_idx:], desc="Processing SpatialScore items", initial=start_idx, total=len(data))):
        actual_idx = i + start_idx  # Calculate the actual index in the full dataset
        
        # Skip if this item has already been processed (as a safeguard)
        if any(r.get('id') == actual_idx for r in all_results):
            continue

        model_inputs = process_model_input(item)
        results = generate_response(model, tokenizer_or_processor, model_inputs)

        # Determine whether response is correct based on question type
        is_correct = False
        question_type = item.get('question_type', '')
        ground_truth = item.get('answer', '')

        if question_type.lower() == 'multi-choice':
            # Extract options for multiple choice questions
            pred_answer = extract_option(results)
            gt_answer = extract_option(ground_truth)
            is_correct = pred_answer.upper() == gt_answer.upper()
        
        elif question_type.lower() == 'judgment':
            # Extract yes/no answers for judgment questions
            pred_answer = extract_yes_no(results)
            gt_answer = extract_yes_no(ground_truth)
            is_correct = pred_answer.lower() == gt_answer.lower()
        
        else:  # Open-ended questions
            if any(unit in ground_truth.lower() for unit in ['meter', 'meters', 'm', 'cm', 'centimeter', 'centimeters', 'km', 'kilometer', 'kilometers', 'inch', 'inches', 'ft', 'foot', 'feet']):
                # Extract numerical value with unit for open-ended questions with units
                is_correct = extract_numeric_with_unit(results, ground_truth)['is_correct']

            elif item.get('source') == 'RealWorldQA':
                is_correct = results.lower() == ground_truth.lower()  # Exact match for RealWorldQA

            else:
                # For other open-ended questions, extract numbers
                try:
                    pred_value = float(extract_number(results))
                except:
                    pred_value = 0.0
                gt_value = float(extract_number(ground_truth))
                
                # Check if both values are extracted successfully
                if pred_value is not None and gt_value is not None:
                    if item.get('source') == 'VSI-Bench_8':
                        is_correct = 'accuracy'
                        if pred_value == 0:
                            score = 1.0 if gt_value == 0.0 else 0.0
                        else:
                            from utils.util import mean_relative_accuracy
                            score = mean_relative_accuracy(pred_value, gt_value, start=0.5, end=0.95, interval=0.05)
                            # ratio = max(pred_value / gt_value, gt_value / pred_value)
                            # is_correct = ratio <= 2.0  # Allow for delta = 2 tolerance
                    else:
                        is_correct = pred_value == gt_value # Exact match for other datasets (SpatialBench, RealWorldQA)
        
        # Update counters
        if is_correct == True:
            total_correct += 1
            score = 1.0
        elif is_correct == False:
            score = 0.0

        total_samples += 1
        
        # Create result entry
        result_entry = {
            "id": item.get('id', actual_idx), "category": item.get('category', 'unknown'), "subcategory": item.get('subcategory', 'unknown'),
            "input_modality": item.get('input_modality', 'image'), "question_type": question_type, "source": item.get('source', 'unknown'), 
            "question": item.get('question', ''), "gt_answer": ground_truth, "pred_answer": results, "img_paths": item.get('img_paths', []), "is_correct": is_correct, "score": score
        }
        
        # Add to all results
        all_results.append(result_entry)
        
        # Group by source
        source = item.get('source', 'unknown')
        if source not in source_results:
            source_results[source] = []
        source_results[source].append(result_entry)
        
        # Group by category
        category = item.get('category', 'unknown')
        if category not in category_results:
            category_results[category] = []
        category_results[category].append(result_entry)
        
        # Save intermediate results periodically
        if (i + 1) % args.save_interval == 0:
            # Save all results so far
            with open(os.path.join(output_dir, f"all_results.json"), 'w') as f:
                json.dump(all_results, f, indent=2)

        torch.cuda.empty_cache()
        
    # Save all results
    with open(os.path.join(output_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save results grouped by source
    for source, results in source_results.items():
        source_dir = os.path.join(output_dir, "by_source")
        os.makedirs(source_dir, exist_ok=True)
        
        with open(os.path.join(source_dir, f"{source}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate and save source-level accuracy using score instead of is_correct
        source_score_sum = sum(r.get('score', 0.0) for r in results)
        source_correct = int(source_score_sum)  # Floor the score as requested
        source_total = len(results)
        source_accuracy = (source_score_sum / source_total) * 100 if source_total > 0 else 0
        
        with open(os.path.join(source_dir, f"{source}_summary.json"), 'w') as f:
            summary = {"source": source, "accuracy": source_accuracy, "correct": source_correct, "total": source_total, "score_sum": source_score_sum}
            json.dump(summary, f, indent=2)
        
        print(f"Source: {source} - Accuracy: {source_accuracy:.2f}% ({source_correct}/{source_total}, score: {source_score_sum:.2f})")

    # Save results grouped by category
    for category, results in category_results.items():
        category_dir = os.path.join(output_dir, "by_category")
        os.makedirs(category_dir, exist_ok=True)
        
        with open(os.path.join(category_dir, f"{category}_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate and save category-level accuracy using score instead of is_correct
        category_score_sum = sum(r.get('score', 0.0) for r in results)
        category_correct = int(category_score_sum)  # Floor the score as requested
        category_total = len(results)
        category_accuracy = (category_score_sum / category_total) * 100 if category_total > 0 else 0
        
        with open(os.path.join(category_dir, f"{category}_summary.json"), 'w') as f:
            summary = {"category": category, "accuracy": category_accuracy, "correct": category_correct, "total": category_total, "score_sum": category_score_sum}
            json.dump(summary, f, indent=2)
        
        print(f"Category: {category} - Accuracy: {category_accuracy:.2f}% ({category_correct}/{category_total}, score: {category_score_sum:.2f})")

    # Calculate and save overall accuracy using score instead of counting is_correct=True
    total_score_sum = sum(r.get('score', 0.0) for r in all_results)
    total_correct = int(total_score_sum)  # Floor the score as requested
    overall_accuracy = (total_score_sum / total_samples) * 100 if total_samples > 0 else 0

    with open(os.path.join(output_dir, "overall_summary.json"), 'w') as f:
        summary = {"accuracy": overall_accuracy, "correct": total_correct, "total": total_samples, "score_sum": total_score_sum}
        json.dump(summary, f, indent=2)

    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples}, score: {total_score_sum:.2f})")
    print(f"All results saved to {output_dir}")

if __name__ == "__main__":
    main()