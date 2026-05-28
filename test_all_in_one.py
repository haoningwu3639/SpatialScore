import os
import json
import torch
import argparse
import logging
import transformers
from tqdm import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info

logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, LlavaForConditionalGeneration, LlavaOnevisionForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from utils.util import image_to_base64_data_uri, extract_number, extract_yes_no, extract_option, extract_numeric_with_unit, load_image

# Optimize GPU memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SEED = 42
torch.manual_seed(SEED)

DATA_ROOT = "./SpatialScore_released/SpatialScore_benchmark"

def load_model_and_components(model_path, model_name="qwen2_5vl-7b"):
    image_processor = None
    if model_name in ["internvl2_5-1b", "internvl2_5-4b", "internvl2_5-8b", "internvl2_5-38b", "internvl2_5-78b", "internvl3-1b", "internvl3-8b", "internvl3-14b", "internvl3-38b", "internvl3-78b", "internvl3_5-1b", "internvl3_5-4b", "internvl3_5-8b", "internvl3_5-14b", "internvl3_5-38b"]:
        # InternVL2_5-1B, InternVL2_5-4B, InternVL2_5-8B, InternVL3-1B, InternVL3-8B, InternVL3-14B, InternVL3-38B, InternVL3-78B, InternVL3.5-1B, InternVL3.5-4B, InternVL3.5-8B, InternVL3.5-14B, InternVL3.5-38B
        model = AutoModel.from_pretrained(model_path, device_map="balanced", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    elif model_name in ["llama", "llama-cot"]:
        # Llama-3.2V-11B-cot
        model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="balanced", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    elif model_name in ["llava1_5-7b", "llava1_5-13b"]:
        model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map="balanced", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    elif model_name in ["llava-ov-7b", "llava-ov-72b"]:
        # LLaVA-OneVision-7B, LLaVA-OneVision-72B
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_path, device_map="balanced", torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    elif model_name in ["qwen2_5vl-3b", "qwen2_5vl-7b", "qwen2_5vl-32b", "qwen2_5vl-72b", "spaceqwen-3b", "spacethinker-qwen2_5vl-3b", "SpaceR"]:
        # Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-72B-Instruct, SpaceQwen-3B-Instruct, QVQ-72B-Preview
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="balanced", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True, min_pixels=256*28*28, max_pixels=2560*28*28)

    elif model_name in ["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "qwen3vl-32b", "qwen3vl-30b-a3b", "qwen3vl-235b-a22b"]:
        # Qwen3VL-2B, Qwen3VL-4B, Qwen3VL-8B, Qwen3VL-32B, Qwen3VL-30B-A3B, Qwen3VL-235B-A22B
        model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="balanced", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2", trust_remote_code=True)
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True, min_pixels=256*28*28, max_pixels=2560*28*28)

    elif model_name in ['kimivl-3b-thinking', 'kimivl-3b']:
        # Kimi-VL-A3B-Thinking, Kimi-VL-A3B-Instruct
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    elif model_name in ["spatialbot-3b"]:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.model.vision_tower.load_model()
        model.model.vision_tower.to(model.device) # This is the bug from SpatialBot itself.

        tokenizer_or_processor = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    elif model_name in ['spacellava']:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        chat_handler = Llava15ChatHandler(clip_model_path=os.path.join(model_path, "mmproj-model-f16.gguf"), verbose=False)
        model = Llama(os.path.join(model_path, "ggml-model-q4_0.gguf"), chat_handler=chat_handler, n_ctx=20480, n_keep=8196, logits_all=True, n_gpu_layers=-1, device_map="balanced", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        # tokenizer_or_processor = None
        tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        
        return model, tokenizer_or_processor, image_processor
    
    model.eval()

    # Simple device map printing
    print(f"\n=== Device Map for {model_name} ===")
    if hasattr(model, 'hf_device_map'):
        print("Device mapping:")
        for name, device in model.hf_device_map.items():
            print(f"  {name}: {device}")
    else:
        print(f"Model device: {next(model.parameters()).device}")
    print("=" * 40)

    return model, tokenizer_or_processor, image_processor

def load_and_process_data(dataset_json_path):
    """Load NDJSON data and process multi-choice questions"""
    data = []
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
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

def process_model_input(sample, model_name="qwen2_5vl-7b"):
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
            clean_rel_path = path[2:] if path.startswith("./") else path
            abs_path = os.path.join(DATA_ROOT, clean_rel_path)
        processed_paths.append(abs_path)

    # Load images based on model type
    if model_name in ["internvl2_5-1b", "internvl2_5-4b", "internvl2_5-8b", "internvl2_5-38b", "internvl2_5-78b", "internvl3-1b", "internvl3-8b", "internvl3-14b", "internvl3-38b", "internvl3-78b", "internvl3_5-1b", "internvl3_5-4b", "internvl3_5-8b", "internvl3_5-14b", "internvl3_5-38b"]:
        images = [load_image(img_path, input_size=448, max_num=32) for img_path in processed_paths]
    else:
        images = [Image.open(path).convert('RGB') for path in processed_paths]
    
    sample['images'] = images
    sample['img_paths'] = processed_paths
    sample['assistant_prompt'] = assistant_prompt

    return sample

def generate_response(model, tokenizer_or_processor, model_inputs, model_name="qwen2_5vl-7b", model_config=None, image_processor=None):
    NUM_BEAMS, TEMPERATURE, MAX_NEW_TOKENS, USE_CACHE = 1, 0.0, 512, True
    DO_SAMPLE = True if TEMPERATURE > 0 else False
    MAX_NEW_TOKENS = 2048 if model_name in ['llama-cot','kimivl-3b-thinking'] else MAX_NEW_TOKENS
    
    if model_name == "spacellava":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = next(model.parameters()).device
    
    assistant_prompt, image_paths, images, text = model_inputs['assistant_prompt'], model_inputs['img_paths'], model_inputs['images'], model_inputs['question']

    with torch.no_grad():        
        if model_name in ["internvl2_5-1b", "internvl2_5-4b", "internvl2_5-8b", "internvl2_5-38b", "internvl2_5-78b", "internvl3-1b", "internvl3-8b", "internvl3-14b", "internvl3-38b", "internvl3-78b", "internvl3_5-1b", "internvl3_5-4b", "internvl3_5-8b", "internvl3_5-14b", "internvl3_5-38b"]:
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

        elif model_name in ["llama", "llama-cot", "llava1_5-7b", "llava1_5-13b", "llava-ov-7b", "llava-ov-72b"]:
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
                
        elif model_name in ["qwen2_5vl-3b", "qwen2_5vl-7b", "qwen2_5vl-32b", "qwen2_5vl-72b", "qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "qwen3vl-32b", "qwen3vl-30b-a3b", "qwen3vl-235b-a22b",  "spaceqwen-3b", "spacethinker-qwen2_5vl-3b", "SpaceR"]:
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
            import io
            import base64
            
            image_content = []
            for image_path in image_paths:
                try:
                    # Load image using PIL and convert to base64
                    with Image.open(image_path) as img:
                        # Ensure image is in RGB format
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save to bytes buffer
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=95)
                        img_bytes = buffer.getvalue()
                        
                        # Create base64 data URI
                        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                        data_uri = f"data:image/jpeg;base64,{img_base64}"
                        
                        image_content.append({
                            "type": "image_url", 
                            "image_url": {"url": data_uri}
                        })
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue
            
            # image_content = [{"type": "image_url", "image_url": {"url": image_to_base64_data_uri(image_path)}} for image_path in image_paths]

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
    parser.add_argument('--model_name', type=str, default='qwen2_5vl-7b', choices=[
                                                                                'qwen2_5vl-3b', 'qwen2_5vl-7b', 'qwen2_5vl-32b', "qwen2_5vl-72b",
                                                                                "qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "qwen3vl-32b", "qwen3vl-30b-a3b", "qwen3vl-235b-a22b",
                                                                                "spaceqwen-3b", "SpaceR", "spacethinker-qwen2_5vl-3b",
                                                                                'llama', 'llama-cot', 'llava-ov-7b', 'llava-ov-72b', 
                                                                                "internvl3_5-1b", "internvl3_5-4b", "internvl3_5-8b", "internvl3_5-14b", "internvl3_5-38b",
                                                                                'internvl3-1b', 'internvl3-8b', 'internvl3-14b', 'internvl3-38b', 'internvl3-78b', 
                                                                                'internvl2_5-1b', 'internvl2_5-4b', 'internvl2_5-8b', 'internvl2_5-38b', 'internvl2_5-78b', 

                                                                                'llava1_5-7b', 'llava1_5-13b', 
                                                                                "kimivl-3b", "kimivl-3b-thinking", "spatialbot-3b", "spacellava",
                                                                               ])
    parser.add_argument('--dataset_json_path', type=str, default="./SpatialScore_benchmark/SpatialScore_benchmark.ndjson")
    parser.add_argument('--output_dir', type=str, default="./eval_results")
    parser.add_argument('--save_interval', type=int, default=50)
    
    args = parser.parse_args()

    model, tokenizer_or_processor, image_processor = load_model_and_components(args.model_path, args.model_name)

    # Load and process NDJSON data
    data = load_and_process_data(args.dataset_json_path)
    
    print(f"Loaded {len(data)} items from unified benchmark")

    # Create output directory structure
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results storage
    all_results = []
    total_samples = 0

    # Check for existing results if resume flag is set
    start_idx = 0
    results_file = os.path.join(output_dir, "all_results.json")
    
    # Resume from existing results if requested
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
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
    for i, item in enumerate(tqdm(data[start_idx:], desc=f"Processing SpatialScore items [{args.model_name}]", initial=start_idx, total=len(data))):
        actual_idx = i + start_idx  # Calculate the actual index in the full dataset
        
        # Skip if this item has already been processed (as a safeguard)
        if any(r.get('id') == item.get('id') for r in all_results):
            continue

        # Process model input and generate response
        model_inputs = process_model_input(item, args.model_name)
        response = generate_response(model, tokenizer_or_processor, model_inputs, args.model_name, model_config=getattr(model, 'config', None), image_processor=image_processor)
        
        # Create result entry without evaluation
        result_entry = {
            "id": item.get('id', actual_idx),
            "source_dataset": item.get('source_dataset', item.get('source', 'unknown')),
            "category": item.get('category', 'unknown'), 
            "task": item.get('task', 'unknown'),
            "sub_task": item.get('sub_task', 'unknown'),
            "input_modality": item.get('input_modality', 'single-image'), 
            "question_type": item.get('question_type', 'unknown'),
            "assistant_prompt": model_inputs['assistant_prompt'],
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
            with open(os.path.join(output_dir, "all_results.json"), 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"Saved intermediate results: {len(all_results)} items processed")

        torch.cuda.empty_cache()
        
    # Save final results
    with open(os.path.join(output_dir, "all_results.json"), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Save a simple summary
    summary = {
        "model_name": args.model_name,
        "dataset_path": args.dataset_json_path,
        "total_processed": len(all_results),
        "total_items": len(data)
    }
    
    with open(os.path.join(output_dir, "processing_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Processing completed: {len(all_results)} items processed out of {len(data)} total items")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()