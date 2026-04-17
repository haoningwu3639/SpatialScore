import os
import json
import torch
import argparse
import transformers
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM
if transformers.__version__ >= "4.49": # Note: cambrian needs transformers == 4.45.0, while others need >= 4.49.0
    from transformers import LlavaForConditionalGeneration, LlavaOnevisionForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText

from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.utils import disable_torch_init
from cambrian.conversation import conv_templates
from cambrian.model.builder import load_pretrained_model
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from utils.util import image_to_base64_data_uri, extract_number, extract_yes_no, extract_option, extract_numeric_with_unit, load_image

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Optimize GPU memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SEED = 42
torch.manual_seed(SEED)

def load_model_and_components(model_path, model_name="qwen2_5vl-7b"):
    image_processor = None
    if model_name == "cambrian":
        disable_torch_init()
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = get_model_name_from_path(model_path)
        tokenizer_or_processor, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device=DEVICE)

    elif model_name in ["internvl2_5-1b", "internvl2_5-4b", "internvl2_5-8b", "internvl2_5-38b", "internvl2_5-78b", "internvl3-1b", "internvl3-8b", "internvl3-14b", "internvl3-38b", "internvl3-78b"]:
        # InternVL2_5-1B, InternVL2_5-4B, InternVL2_5-8B, InternVL3-1B, InternVL3-8B, InternVL3-14B, InternVL3-38B, InternVL3-78B
        model = AutoModel.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    elif model_name in ["llama", "llama-cot"]:
        # Llama-3.2V-11B-cot
        model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    elif model_name in ["llava1_5-13b"]:
        model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    elif model_name in ["llava-ov-7b", "llava-ov-72b"]:
        # LLaVA-OneVision-7B, LLaVA-OneVision-72B
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    elif model_name in ["qwen2_5vl-3b", "qwen2_5vl-7b", "qwen2_5vl-32b", "qwen2_5vl-72b", "spaceqwen-3b", "spacethinker-qwen2_5vl-3b", "SpaceR"]:
        # Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-72B-Instruct, SpaceQwen-3B-Instruct, QVQ-72B-Preview
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True, min_pixels=256*28*28, max_pixels=2560*28*28)

    elif model_name in ['kimivl-3b-thinking', 'kimivl-3b']:
        # Kimi-VL-A3B-Thinking, Kimi-VL-A3B-Instruct
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    elif model_name in ["spatialbot-3b"]:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.model.vision_tower.load_model()
        model.model.vision_tower.to(model.device) # This is the bug from SpatialBot itself.

        tokenizer_or_processor = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    elif model_name in ['spacellava']:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        chat_handler = Llava15ChatHandler(clip_model_path=os.path.join(model_path, "mmproj-model-f16.gguf"), verbose=False)
        model = Llama(os.path.join(model_path, "ggml-model-q4_0.gguf"), chat_handler=chat_handler, n_ctx=15360, n_keep=8196, logits_all=True, n_gpu_layers=-1, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # tokenizer_or_processor = None
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        
        return model, tokenizer_or_processor, image_processor
    
    model.eval()

    return model, tokenizer_or_processor, image_processor

def process_model_input(sample, model_name="qwen2_5vl-7b"):
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
    if model_name in ["internvl2_5-1b", "internvl2_5-4b", "internvl2_5-8b", "internvl2_5-38b", "internvl2_5-78b", "internvl3-1b", "internvl3-8b", "internvl3-14b", "internvl3-38b", "internvl3-78b"]:
        images = [load_image(img_path, max_num=32) for img_path in sample['img_paths']] # max_num=128 is the maximum patches for InternVL
    else: # For other models, we can use PIL to load images
        images = [Image.open(img_path).convert('RGB') for img_path in sample['img_paths']]
    
    sample['images'] = images
    sample['assistant_prompt'] = assistant_prompt

    return sample

def generate_response(model, tokenizer_or_processor, model_inputs, model_name="qwen2_5vl-7b", model_config=None, image_processor=None):
    NUM_BEAMS, TEMPERATURE, MAX_NEW_TOKENS, USE_CACHE = 1, 0.0, 1024, True
    DO_SAMPLE = True if TEMPERATURE > 0 else False
    MAX_NEW_TOKENS = 2048 if model_name == 'kimivl-3b-thinking' else MAX_NEW_TOKENS
    
    device = next(model.parameters()).device
    if model_name == "spacellava":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = next(model.parameters()).device
    
    assistant_prompt, image_paths, images, text = model_inputs['assistant_prompt'], model_inputs['img_paths'], model_inputs['images'], model_inputs['question']

    with torch.no_grad():
        if model_name == "cambrian":
            question = assistant_prompt + text
            # Multi-image for multi-encoder in Cambrian
            if model_config.mm_use_im_start_end:
                input_text = ''.join([DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN for _ in images]) + '\n' + question
            else:
                input_text = ''.join([DEFAULT_IMAGE_TOKEN for _ in images]) + '\n' + question

            conv = conv_templates["llama_3"].copy()
            conv.append_message(conv.roles[0], input_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image_sizes = [img.size for img in images]
            image_tensor = process_images(images, image_processor, model_config)
            image_tensor = [tensor.to(device) for tensor in image_tensor if tensor.numel() > 0]

            input_ids = tokenizer_image_token(prompt, tokenizer_or_processor, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            output_ids = model.generate(input_ids, images=image_tensor, image_sizes=image_sizes, num_beams=NUM_BEAMS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, use_cache=USE_CACHE, do_sample=DO_SAMPLE)
            response = tokenizer_or_processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            prompt_end_idx = response.find('\n')
            if prompt_end_idx != -1:
                response = response[prompt_end_idx+1:].strip()
                
        elif model_name in ["internvl2_5-1b", "internvl2_5-4b", "internvl2_5-8b", "internvl2_5-38b", "internvl2_5-78b", "internvl3-1b", "internvl3-8b", "internvl3-14b", "internvl3-38b", "internvl3-78b"]:
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

        elif model_name in ["llama", "llama-cot", "llava1_5-13b", "llava-ov-7b", "llava-ov-72b"]:
            image_content = []
            # Process images to ensure proper sizes
            for i in range(len(images)):
                if images[i].size[0] <= 3 or images[i].size[1] <= 3:
                    images[i] = images[i].resize((32, 32), Image.BICUBIC)
            for _ in range(len(images)):
                image_content.append({'type': 'image'}) # Here, just add <image> special token

            messages = [
                {'role': 'assistant', 'content': [{'type': 'text', 'text': assistant_prompt}]},                
                {'role': 'user', 'content': image_content + [{'type': 'text', 'text': text}]}
            ]

            input_text = tokenizer_or_processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer_or_processor(images, input_text, return_tensors='pt').to(device) # process images and text here
            output = model.generate(**inputs, num_beams=NUM_BEAMS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, use_cache=USE_CACHE, do_sample=DO_SAMPLE)
            response = tokenizer_or_processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            if "ASSISTANT" in response:
                response = response.split("ASSISTANT")[-1].strip()
                
        elif model_name in ["qwen2_5vl-3b", "qwen2_5vl-7b", "qwen2_5vl-32b", "qwen2_5vl-72b", "spaceqwen-3b", "spacethinker-qwen2_5vl-3b", "SpaceR"]:
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
        
        elif model_name in ["kimivl-3b-thinking", "kimivl-3b"]:
            image_content = [{"type": "image", "image": image} for image in images]
            messages = [
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": image_content + [{"type": "text", "text": text}]}
            ]
            input_text = tokenizer_or_processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            inputs = tokenizer_or_processor(images=images, text=input_text, return_tensors="pt", padding=True).to(device)
            
            output_ids = model.generate(**inputs, num_beams=NUM_BEAMS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, use_cache=USE_CACHE, do_sample=DO_SAMPLE)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            response = tokenizer_or_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        elif model_name in ["spatialbot-3b"]:
            # Make sure the model is on the correct device

            # Format text with special token placeholders for images
            image_placeholders = "\n".join([f"<image {i+1}>" for i in range(len(images))])
            prompt_text = f"{assistant_prompt} USER: {image_placeholders}\n{text} ASSISTANT:"
            
            # Split and tokenize the text around image placeholders
            text_chunks = [tokenizer_or_processor(chunk).input_ids for chunk in prompt_text.split(image_placeholders)]
            
            # Create input_ids with special image tokens
            offset_bos = 0  # Adjust if needed
            input_ids_list = text_chunks[0]
            for i in range(len(images)):
                input_ids_list += [-(201 + i)]  # Special tokens for each image
            input_ids_list += text_chunks[1][offset_bos:]
            
            input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
            
            # Process images
            image_tensor = model.process_images(images, model.config).to(dtype=model.dtype, device=device)
            
            # Generate response
            output_ids = model.generate(input_ids, images=image_tensor, repetition_penalty=1.0, num_beams=NUM_BEAMS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, use_cache=USE_CACHE, do_sample=DO_SAMPLE)[0]
            
            # Decode response
            response = tokenizer_or_processor.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

        elif model_name in ["spacellava"]:
            image_content = [{"type": "image_url", "image_url": {"url": image_to_base64_data_uri(image_path)}} for image_path in image_paths]
            messages = [
                {"role": "system", "content": assistant_prompt},
                {"role": "user", "content": image_content + [{"type": "text", "text": text}]}
            ]

            result = model.create_chat_completion(messages=messages, temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS, stream=False)
            response = result["choices"][0]["message"]["content"]

    torch.cuda.empty_cache()
    
    return response

def main():
    parser = argparse.ArgumentParser(description="SpatialScore Evaluation")
    parser.add_argument('--model_path', type=str, default="./huggingface/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--model_name', type=str, default='qwen2_5vl-7b', choices=['cambrian', 'internvl2_5-1b', 'internvl2_5-4b', 'internvl2_5-8b', 'internvl2_5-38b', 'internvl2_5-78b', 
                                                                               'internvl3-1b', 'internvl3-8b', 'internvl3-14b', 'internvl3-38b', 'internvl3-78b', 
                                                                               'llama', 'llama-cot', 'llava1_5-13b', 'llava-ov-7b', 'llava-ov-72b', 'qwen2_5vl-3b', 'qwen2_5vl-7b', 'qwen2_5vl-32b', "qwen2_5vl-72b", "qvq-72b",
                                                                               "spatialbot-3b", "spacellava", "spaceqwen-3b", "kimivl-3b", "kimivl-3b-thinking", "spacethinker-qwen2_5vl-3b", "SpaceR"
                                                                               ])
    parser.add_argument('--dataset_json_path', type=str, default="./dataset/SpatialScore.json")
    parser.add_argument('--dataset_name', type=str, default="all", choices=['all', 'VGBench', 'cvbench', 'spatialbench', 'MMIU', 'BLINK', '3DSRBench', 'RealWorldQA', 'MMVP', 'QSpatialBench-Plus', 'QSpatialBench-ScanNet', 'VSR-ZeroShot', 'SpatialSense', 'VSI-Bench_8'])
    parser.add_argument('--output_dir', type=str, default="./eval_results")
    parser.add_argument('--save_interval', type=int, default=20)
    
    args = parser.parse_args()

    model, tokenizer_or_processor, image_processor = load_model_and_components(args.model_path, args.model_name)

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

        model_inputs = process_model_input(item, args.model_name)
        results = generate_response(model, tokenizer_or_processor, model_inputs, args.model_name, model_config=getattr(model, 'config', None), image_processor=image_processor)

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