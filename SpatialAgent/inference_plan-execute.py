import os
import re
import json
import torch
import argparse
import traceback
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, AutoModel

from data import AgentDataset
from action import (
                    SelfThinking, Detect3DObjects,
                    LocalizeObjects, CountObjects, GetObjectMask, EstimateRegionDepth, 
                    EstimateObjectDepth, GetCameraParametersVGGT, EstimateObjectGeometryProperties, 
                    MatchImagesSIFT, EstimateHomographyMatrix, EstimateOpticalFlow, GetObjectOrientation, 
                    Terminate, Get3DDistance, EstimateImageDepth
                )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help="random seed.")
    parser.add_argument('--prompt_format', default='cota', type=str, help="the inference prompt format", choices=['cota', 'direct'])
    parser.add_argument('--data_path', type=str, default="../SpatialScore_benchmark/SpatialScore_benchmark.ndjson")
    parser.add_argument('--model_path', type=str, default="../huggingface/Qwen3-VL-4B-Instruct")
    parser.add_argument('--model_name', type=str, default='qwen3vl-4b', choices=['qwen3vl-2b', 'qwen3vl-4b', "qwen3vl-8b", "qwen3vl-32b"])
    parser.add_argument('--output_dir', default='./output-pe', type=str, help="output directory.")
    parser.add_argument('--exp_id', default="", type=str, help="unique experiment ID.")

        
    parser.add_argument('--start', type=int, default=0, help="start index of dataset to process")
    parser.add_argument('--end', type=int, default=-1, help="end index of dataset to process (inclusive)")

    args = parser.parse_args()

    return args

def create_action_registry(model=None, processor=None, depth_estimator=None):
    """Create a registry of action instances to avoid reinitializing tools"""
    registry = {}

    registry["LocalizeObjects"] = LocalizeObjects()
    registry["CountObjects"] = CountObjects()
    registry["GetObjectMask"] = GetObjectMask()

    registry["EstimateOpticalFlow"] = EstimateOpticalFlow()
    registry["MatchImagesSIFT"] = MatchImagesSIFT()
    registry["EstimateHomographyMatrix"] = EstimateHomographyMatrix()

    registry["EstimateRegionDepth"] = EstimateRegionDepth(depth_estimator=depth_estimator)
    registry["EstimateObjectDepth"] = EstimateObjectDepth(object_localizer=registry["LocalizeObjects"], depth_estimator=depth_estimator)
    registry["GetCameraParametersVGGT"] = GetCameraParametersVGGT()
    registry["EstimateObjectGeometryProperties"] = EstimateObjectGeometryProperties(mask_generator=registry["GetObjectMask"], depth_estimator=registry["EstimateObjectDepth"], camera_params_estimator=registry["GetCameraParametersVGGT"])
    registry["GetObjectOrientation"] = GetObjectOrientation(localizer=registry["LocalizeObjects"])
    registry["Get3DDistance"] = Get3DDistance()

    registry["SelfThinking"] = SelfThinking(model=model, processor=processor)
    registry["Terminate"] = Terminate()

    registry["Detect3DObjects"] = Detect3DObjects()

    return registry

def generate_action_details(actions):
    details = []
    
    for action in actions:
        action_info = f"Tool: {action.name}\n"
        action_info += f"Description: {action.description}\n"
        action_info += "Arguments:\n"
        for arg_name, arg_desc in action.args_spec.items():
            action_info += f"  - {arg_name}: {arg_desc}\n"
        action_info += "Returns:\n"
        for ret_name, ret_desc in action.rets_spec.items():
            action_info += f"  - {ret_name}: {ret_desc}\n"
        action_info += "Examples:\n"
        for example in action.examples:
            example_copy = example.copy()
            if 'arguments' in example_copy and 'image' in example_copy['arguments']:
                example_copy['arguments']['image'] = "image-0"
            action_info += f"  - {json.dumps(example_copy)}\n"
        details.append(action_info)

    return "\n".join(details)

def process_vision_inputs(messages):
    image_inputs, video_inputs = [], []

    for msg in messages:
        if isinstance(msg["content"], list):
            for content in msg["content"]:
                if content.get("type") == "image":
                    image_path = content["image"]
                    image = Image.open(image_path).convert("RGB")
                    image_inputs.append(image)
    
    return image_inputs, video_inputs

def generate_qwen_response(model, processor, messages):
    NUM_BEAMS, TEMPERATURE, MAX_NEW_TOKENS, USE_CACHE = 1, 0.0, 4096, True
    DO_SAMPLE = True if TEMPERATURE > 0 else False
    device = next(model.parameters()).device

    input_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_inputs(messages)
    inputs = processor(
        text=[input_text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, num_beams=NUM_BEAMS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, use_cache=USE_CACHE, do_sample=DO_SAMPLE)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    
    torch.cuda.empty_cache()
    
    return response

def execute_tool_plan(plan, actions, task, max_retries=3):
    results = {}
    action_map = {action.name: action for action in actions} # obtain all actions
    image_paths = task.get('image_paths', [])
    
    def get_image_path(image_id_str):
        if image_id_str.startswith("image-"): # image-0
            image_idx = int(image_id_str.split("-")[1]) # 0, 1, 2
            if image_idx < len(image_paths):
                path = image_paths[image_idx]
                return path
        elif isinstance(image_id_str, str) and os.path.isabs(image_id_str):
            if os.path.exists(image_id_str):
                return image_id_str
    
    for step in plan:
        tool_name, args = step["name"], step["arguments"].copy()
        retries, success, result = 0, False, None

        while retries < max_retries and not success:
            try:
                action = action_map.get(tool_name)
                if action is None:
                    raise ValueError(f"Unknown tool: {tool_name}")
                # Handle 'image' field that could be a string or a list
                if 'image' in args:
                    if isinstance(args['image'], list):
                        # Handle a list of image identifiers
                        processed_images = []
                        for img_id in args['image']:
                            img_path = get_image_path(img_id)
                            if img_path:
                                processed_images.append(img_path)
                        args['image'] = processed_images if processed_images else args['image']
                    else:
                        # Handle a single image identifier
                        img_path = get_image_path(args['image'])
                        if img_path:
                            args['image'] = img_path

                if 'images' in args:
                    if isinstance(args['images'], list):
                        # Handle a list of image identifiers
                        processed_images = []
                        for img_id in args['images']:
                            img_path = get_image_path(img_id)
                            if img_path:
                                processed_images.append(img_path)
                        args['images'] = processed_images if processed_images else args['images']
                    else:
                        # Handle a single image identifier
                        img_path = get_image_path(args['images'])
                        if img_path:
                            args['images'] = img_path
                if 'other_images' in args:
                    args['other_images'] = [get_image_path(img) for img in args['other_images']]

                result = action(**args)
                success = True
                results[tool_name] = result
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    results[tool_name] = f"Failed after {max_retries} retries: {str(e)}"
                    break
    return results

def parse_cot_response(cot_response, tool_results):
    final_thinking, answer = "", ""
    cot_steps, current_step = [], {}
    
    # Extract thinking-tool-observation triplets
    response_text = cot_response
    
    # Process each thinking block
    thinking_blocks = re.findall(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL | re.IGNORECASE)
    for i, thinking in enumerate(thinking_blocks):
        current_step = {"thinking": thinking.strip()}
        
        # Look for the matching tool after this thinking block
        tool_match_pattern = r'<thinking>' + re.escape(thinking) + r'</thinking>.*?<tool>(.*?)</tool>'
        tool_match = re.search(tool_match_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if tool_match:
            tool_content = tool_match.group(1).strip()
            try:
                current_step["tool"] = json.loads(tool_content)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool JSON: {tool_content}, error: {e}")
                current_step["tool"] = {"raw": tool_content}
                
            # Look for the matching observation after this tool
            obs_pattern = r'<tool>' + re.escape(tool_content) + r'</tool>.*?<observation>(.*?)</observation>'
            obs_match = re.search(obs_pattern, response_text, re.DOTALL | re.IGNORECASE)
            if obs_match:
                observation_content = obs_match.group(1).strip()
                try:
                    current_step["observation"] = json.loads(observation_content)
                except json.JSONDecodeError:
                    current_step["observation"] = observation_content
                
                # Validate and update observation using tool_results
                if "tool" in current_step and isinstance(current_step["tool"], dict):
                    tool_name = current_step["tool"].get("name", "")
                    if tool_name in tool_results:
                        expected_observation = tool_results[tool_name]
                        current_step["observation"] = expected_observation
        
        # Add the completed step to cot_steps
        if current_step.get("thinking"):
            cot_steps.append(current_step)
    
    # Extract final_thinking and answer if present
    final_thinking_match = re.search(r'<final_thinking>(.*?)(?:</final_thinking>|$)', cot_response, re.DOTALL | re.IGNORECASE)
    if final_thinking_match:
        final_thinking = final_thinking_match.group(1).strip()
    
    answer_match = re.search(r'<answer>(.*?)(?:</answer>|$)', cot_response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return cot_steps, final_thinking, answer

def parse_final_response(final_response):
    final_thinking, answer = "", ""

    # Extract thinking content - first try with tags
    thinking_match = re.search(r'<thinking>(.*?)(?:</thinking>|$)', final_response, re.DOTALL | re.IGNORECASE)
    if thinking_match:
        final_thinking = thinking_match.group(1).strip()
    else:
        # Try to extract thinking from other formats
        thinking_match = re.search(r'(?:Thinking:|My thinking:|Reasoning:)(.*?)(?:<answer>|Answer:|My answer:|$)', final_response, re.DOTALL | re.IGNORECASE)
        if thinking_match:
            final_thinking = thinking_match.group(1).strip()
    
    # Extract answer content - first try with tags
    answer_match = re.search(r'<answer>(.*?)(?:</answer>|$)', final_response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Try to extract answer from other formats
        answer_match = re.search(r'(?:Answer:|My answer:|The answer is:)(.*?)$', final_response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
    
    # If we still couldn't find an answer but there's a line with "**" (bold markdown) which often indicates the final answer
    if not answer:
        bold_match = re.search(r'\*\*(.*?)\*\*', final_response, re.DOTALL)
        if bold_match:
            answer = bold_match.group(0).strip()  # Include the ** markers
    
    return final_thinking, answer

def main():
    args = get_args()

    run_id = f"{args.exp_id if args.exp_id else ''}-{args.prompt_format}-seed-{args.seed}-{str(args.start)}-{str(args.end)}".replace('/', '-')
    output_dir = os.path.join(args.output_dir, args.model_name)
    full_result_path = os.path.join(args.output_dir, args.model_name, run_id)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(full_result_path, exist_ok=True)

    # Load previously processed examples
    wf_name = os.path.join(output_dir, f"{run_id}.jsonl")
    has_inferenced = {}
    if os.path.exists(wf_name):
        with open(wf_name, "r") as rf:
            for line in rf:
                saved_data = json.loads(line)
                has_inferenced[saved_data["id"]] = saved_data
    wf = open(wf_name, "a")

    # Initialize all_results list for unified output
    all_results_file = os.path.join(output_dir, f"{run_id}_all_results.json")
    all_results = []

    # Check if we're resuming and load existing results
    if os.path.exists(all_results_file):
        with open(all_results_file, "r", encoding='utf-8') as f:
            all_results = json.load(f)

    if args.model_name in ["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "qwen3vl-32b"]:
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True, min_pixels=256*28*28, max_pixels=2560*28*28)

    # Create actions with shared tool instances
    depth_estimator = EstimateImageDepth()
    action_registry = create_action_registry(model, processor, depth_estimator)
    actions = list(action_registry.values())

    action_details = generate_action_details(actions)
    
    print("Successful Tool Initialization.")

    # 1. planning_prompt
    planning_prompt = (
        """[BEGIN OF GOAL]\n"""
        """Generate a JSON-formatted tool-calling plan to solve spatial understanding questions about given images or videos.\n"""
        """[END OF GOAL]\n\n"""
        """[BEGIN OF TOOLBOX]\n"""
        f"{action_details}\n"
        """[END OF TOOLBOX]\n\n"""
        """[BEGIN OF TASK INSTRUCTIONS]\n"""
        """Generate a step-by-step plan to answer the given spatial understanding question about given images or videos. """
        """***Use ONLY the tools listed in the TOOLBOX section (e.g., GetObjectOrientation, EstimateObjectGeometryProperties, LocalizeObjects, EstimateObjectDepth)***"""
        """***Follow their argument specifications EXACTLY as defined in the toolbox, and try to give detailed and comprehensive instructions in queries.*** """
        """Do NOT invent new tools or modify the existing tool interfaces. """
        """The plan should strictly follow what these tools can and cannot do.\n"""
        """[END OF TASK INSTRUCTIONS]\n\n"""
        """[BEGIN OF FORMAT INSTRUCTIONS]\n"""
        """You are a helpful assistant tasked with solving spatial reasoning questions. Think step by step. """
        """***"""
        """Return a JSON list of tool calls inside ```json``` tags, where each call is a dictionary with 'name' and 'arguments'. """
        """The 'name' MUST match exactly one of the tool names provided in the toolbox. """
        """The 'arguments' MUST include ALL required parameters for that specific tool with EXACT parameter names. """
        """The 'images' or 'image' argument must be specified as 'image-0', 'image-1', and 'image-2', to refer to the provided images. """
        """Do not answer the question directly, and do not use absolute paths for the 'images' or 'image' argument."""
        """***\n"""
        """[END OF FORMAT INSTRUCTIONS]\n\n"""
        """[BEGIN OF EXAMPLES]\n"""
        """Example for 'Which is closer to the camera, the dog or the cat?':\n"""
        """```json\n"""
        """[\n"""
        """  {"name": "LocalizeObjects", "arguments": {"image": "image-0", "objects": ["dog", "cat"]}},\n"""
        """  {"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "objects": ["dog", "cat"], "indoor_or_outdoor": "outdoor"}},\n"""
        """]\n"""
        """```\n"""
        """[END OF EXAMPLES]\n\n"""
        """***"""
        """Do not answer the question directly. Instead, think step-by-step, and output the tool-calling plan inside ```json``` tags."""
        """***\n"""
    )

    # 2. cot_prompt
    cot_prompt = (
        """[BEGIN OF GOAL]\n"""
        """Generate a Chain of Thought (CoT) reasoning process using the provided tool execution results.\n"""
        """[END OF GOAL]\n\n"""
        """[BEGIN OF TASK INSTRUCTIONS]\n"""
        """You are a helpful assistant tasked with solving spatial reasoning questions. """
        """Analyze the given question and tool execution results. Think step by step. """
        """Generate a step-by-step reasoning process that shows how the tools contribute to solving the question. """
        """Use ONLY the tools and results provided, following their specifications STRICTLY.\n"""
        """The results of tool calls can sometimes be incomplete or incorrect, so please be critical and decide how to make use of them. """
        """If a tool failed, note the failure and proceed with your prior knowledge and reasoning.\n"""
        """Repeat for each tool result in order. \n"""
        """[END OF TASK INSTRUCTIONS]\n\n"""
        """[BEGIN OF FORMAT INSTRUCTIONS]\n"""
        """***\n"""
        """Output a CoT with:\n"""
        """- <thinking> Explain why this tool was used and how its result contributes to the answer. </thinking>\n"""
        """- <tool> The tool call in JSON format, e.g., {{"name": "LocalizeObjects", "arguments": {{"image": "image-0", "objects": ["dog", "cat"]}}}}. </tool>\n"""
        """- <observation>: The tool result as a string. </observation>\n"""
        """Repeat for each tool result in order.\n"""
        """***\n"""
        """[END OF FORMAT INSTRUCTIONS]\n\n"""
        """[BEGIN OF EXAMPLES]\n"""
        """Example for 'In image-0, which is closer to the camera, the dog or the cat?':\n"""
        """<thinking> To determine which object is closer to the camera, I need first localize the dog and cat in the image. </thinking>\n"""
        """<tool> {{"name": "LocalizeObjects", "arguments": {{"image": "image-0", "objects": ["dog", "cat"]}}}} </tool>\n"""
        """<observation> {{"results": [{{"label": "dog", "region": [0.5, 0.6, 0.6, 0.8], "confidence": 0.95}}, {{"label": "cat", "region": [0.4, 0.5, 0.45, 0.7], "confidence": 0.87}}]}} </observation>\n"""
        """<thinking> The bounding box for the dog is [0.5, 0.6, 0.6, 0.8], and for the cat is [0.4, 0.5, 0.45, 0.7]. Then, I need estimate the depth of them to reflect their distances to the camera. </thinking>\n"""
        """<tool> {{"name": "EstimateObjectDepth", "arguments": {{"image": "image-0", "objects": ["dog", "cat"], "indoor_or_outdoor": "outdoor"}}}} </tool>\n"""
        """<observation> {{"results": [{{"object": "dog", "depth": 1.0, "error": null}}, {{"object": "cat", "depth": 1.2, "error": null}}]}} </observation>\n"""
        """[END OF EXAMPLES]\n\n"""
        """Tool Plan: \n{tool_plan}\n"""
        """Tool Results: \n{tool_results}\n"""
        """***\n"""
        """**Notably, you should AVOID outputting terms like <final_thining>, <answer>, or <final_answer> here.** """
        """**Now, output your reasoning between <thinking> and </thinking>, the tool call in JSON format between <tool> and </tool>, and the observation between <observation> and </observation>.**\n"""
        """***\n"""
    )

    # 3. final_prompt
    final_prompt = (
        """[BEGIN OF GOAL]\n"""
        """Generate a final REASONING and ANSWER for spatial understanding questions about given images or videos, based on tool results and prior Chain of Thought (CoT) steps.\n"""
        """[END OF GOAL]\n\n"""
        """[BEGIN OF TASK INSTRUCTIONS]\n"""
        """You are a helpful assistant tasked with solving spatial reasoning questions. """
        """Given the question, tool execution results, and CoT steps, synthesize the information to provide a final REASONING and ANSWER. """
        """**The results of tool calls can sometimes be incomplete or incorrect, so please be critical and decide how to make use of them.** """
        """If tool results are unclear or contradictory, use your prior knowledge to think the problem step-by-step.\n"""
        """For multi-choice questions, select the most appropriate answer from options based on reasoning. Respond ONLY with the capital letter and its parentheses.\n"""
        """For judgment questions, answer with yes or no based on reasoning. Respond ONLY with 'yes' or 'no'.\n"""
        """For open-ended measurement questions, answer the question by measuring the precise distance in 3D space through a 2D images or videos. DO NOT use generic and unclear units like 'units' or 'pixels'"""
        """Respond ONLY with a numeric answer consisting of a scalar and a distance unit in the format of **\\scalar \\distance_unit**."""
        """For other questions, answer the question based on the given image or video. Respond ONLY with a concise and accurate scalar or a scalar with corresponding unit.\n"""
        """**CRITICAL: You MUST always provide a reasonable answer. Never respond with 'cannot be determined', 'none of the above', or similar phrases.**\n"""
        """[END OF TASK INSTRUCTIONS]\n\n"""
        """[BEGIN OF FORMAT INSTRUCTIONS]\n"""
        """***\n"""
        """Output:\n"""
        """- <thinking> A complete analysis synthesizing all tool results and CoT steps to derive the answer. </thinking>\n"""
        """- <answer> **The final answer** </answer>\n"""
        """***\n"""
        """[END OF FORMAT INSTRUCTIONS]\n\n"""
        """CoT Steps: \n{cot_steps}\n"""
        """***\n"""
        """**CRITICAL: You MUST always provide a reasonable answer. Never respond with 'cannot be determined', 'none of the above', or similar phrases.**\n"""
        """Now, output **your thinking** between <thinking> and </thinking>, and **your answer** between <answer> and </answer>.\n"""
        """***\n"""
    )

    # 4. direct_prompt
    direct_prompt = (
        """[BEGIN OF GOAL]\n"""
        """Provide a direct ANSWER to a spatial understanding questions about given 2D images or videos without external tools.\n"""
        """[END OF GOAL]\n\n"""
        """[BEGIN OF TASK INSTRUCTIONS]\n"""
        """You are a helpful assistant tasked with solving spatial reasoning questions. Think step by step. """
        """Answer the spatial understanding question by reasoning about the provided images or videos. """
        """Provide a direct answer by reasoning logically based on typical spatial relationships and visual cues in the images or videos.\n"""
        """**CRITICAL: You MUST always provide a reasonable answer. Never respond with 'cannot be determined', 'none of the above', or similar phrases.**\n"""
        """***"""
        """For multi-choice questions, select the most appropriate answer from options based on reasoning. Respond ONLY with the capital letter and its parentheses.\n"""
        """For judgment questions, answer with yes or no based on reasoning. Respond ONLY with 'yes' or 'no'.\n"""
        """For open-ended measurement questions, answer the question by measuring the precise distance in 3D space through a 2D images or videos. DO NOT use generic and unclear units like 'units' or 'pixels'"""
        """Respond ONLY with a numeric answer consisting of a scalar and a distance unit in the format of **\\scalar \\distance_unit**."""
        """For other questions, answer the question based on the given image or video. Respond ONLY with a concise and accurate scalar or a scalar with corresponding unit.\n"""
        """***\n"""
        """[END OF TASK INSTRUCTIONS]\n\n"""
        """[BEGIN OF FORMAT INSTRUCTIONS]\n"""
        """***\n"""
        """Output your response in the format:\n"""
        """- <thinking> [Your reasoning here] </thinking>\n"""
        """- <answer> [Your final answer] </answer>\n"""
        """***\n"""
        """[END OF FORMAT INSTRUCTIONS]\n\n"""
        """***\n"""
        """**CRITICAL: You MUST always provide a reasonable answer. Never respond with 'cannot be determined', 'none of the above', or similar phrases.**\n"""
        """Now, output **your thinking** between <thinking> and </thinking>, and **your answer** between <answer> and </answer>.\n"""
        """***\n"""
    )
    
    data = AgentDataset(dataset_json_path=args.data_path)

    print("Successful Data Initialization")

    if args.end == -1:
        args.end = len(data) - 1

    for idx in range(args.start, args.end + 1):
        try:
            sample = data[idx]
            # Skip if sample ID is in has_inferenced
            if 'id' in sample and sample['id'] in has_inferenced:
                print(f"Skipping already processed sample with ID: {sample['id']}")
                continue
                
            print(f"Processing sample with ID: {sample['id']}: {sample}")
            
            image_content = [{"type": "image", "image": img_path} for img_path in sample['image_paths']]
            messages = [
                    {"role": "assistant", "content": planning_prompt},
                    {"role": "user", "content": image_content + [{"type": "text", "text": sample['assistant_prompt'] + sample['question']}]}
                ]
            
            # 1. Generate tool plan
            plan_response = generate_qwen_response(model, processor, messages)
            # Note: plan will be in a format like: ``` json\n[{"name": "GetImageToImagesSimilarity", "arguments": {"image": "image-0", "other_images": ["image-1"]}}]\n```
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', plan_response, re.IGNORECASE)
            
            if json_match:
                json_str = json_match.group(1)
                tool_plan = json.loads(json_str)
            else:
                print(plan_response)
                with open(os.path.join(output_dir, "skipped.log"), "a") as log:
                    log.write(f"{idx}: cannot find JSON code block\n")
                continue
            
            for step in tool_plan:
                if "arguments" in step and "image" in step["arguments"]:
                    image_arg = step["arguments"]["image"]
                    # Handle both string and list formats
                    if isinstance(image_arg, list):
                        # Check each image identifier in the list
                        for img in image_arg:
                            if not img.startswith("image-"):
                                with open(os.path.join(output_dir, "plan_error.log"), "a") as log:
                                    log.write(f"Example {idx}: Tool plan used non image-0 format in list: {img}\n")
                    elif not image_arg.startswith("image-"):
                        # Handle single image identifier
                        with open(os.path.join(output_dir, "plan_error.log"), "a") as log:
                            log.write(f"Example {idx}: Tool plan used non image-0 format: {image_arg}\n")
            
            print(f"tool plan: {tool_plan}")
            # 2. Execute tool plan
            tool_results = execute_tool_plan(tool_plan, actions, sample)
            print(f"tool results: {tool_results}")

            all_failed = all("Failed" in str(result) for result in tool_results.values()) and tool_results
            # If all tools failed, we need to use the direct mode
            if all_failed:
                final_query = direct_prompt
                messages = [
                        {"role": "assistant", "content": final_query},
                        {"role": "user", "content": image_content + [{"type": "text", "text": sample['assistant_prompt'] + sample['question']}]}
                    ]
                final_answer = generate_qwen_response(model, processor, messages)
                cot_steps, final_thinking, answer = [], "", ""

                print(f"Direct Mode Response: {final_answer}")
                # Use regex with re.DOTALL to handle multi-line tags
                thinking_match = re.search(r'<thinking>(.*?)(?:</thinking>|$)', final_answer, re.DOTALL | re.IGNORECASE)
                if thinking_match:
                    final_thinking = thinking_match.group(1).strip()

                answer_match = re.search(r'<answer>(.*?)(?:</answer>|$)', final_answer, re.DOTALL | re.IGNORECASE)
                if answer_match:
                    answer = answer_match.group(1).strip()
            
            else:
                # 3. Generate CoT steps
                cot_query = cot_prompt.format(tool_plan=json.dumps(tool_plan, indent=4), tool_results=json.dumps(tool_results, indent=4))
                messages = [
                        {"role": "assistant", "content": cot_query},
                        # {"role": "user", "content": image_content + [{"type": "text", "text": sample['assistant_prompt'] + sample['question']}]}
                        {"role": "user", "content": image_content + [{"type": "text", "text": sample['question']}]}
                    ]
                cot_response = generate_qwen_response(model, processor, messages)
                print(f"CoT Response: {cot_response}")
                cot_steps, _, _ = parse_cot_response(cot_response, tool_results)

                # 4. Format CoT steps for final query
                cot_steps_str = ""
                for step in cot_steps:
                    if "thinking" in step.keys():
                        cot_steps_str += f"<thinking>{step['thinking']}</thinking>\n"
                    if "tool" in step.keys():
                        cot_steps_str += f"<tool>{json.dumps(step['tool'])}</tool>\n"
                    if "observation" in step.keys():
                        cot_steps_str += f"<observation>{json.dumps(step['observation'])}</observation>\n"

                print(f"CoT Steps: {cot_steps_str}")

                # 5. Generate final reasoning and answer
                final_query = final_prompt.format(cot_steps=cot_steps_str)
                    
                messages = [
                    {"role": "assistant", "content": final_query},
                    {"role": "user", "content": image_content + [{"type": "text", "text": sample['assistant_prompt'] + sample['question']}]}
                ]
                final_response = generate_qwen_response(model, processor, messages)
                print(f"Final Response: {final_response}")

                final_thinking, answer = parse_final_response(final_response)

            # Fallback mechanism    
            if not answer or answer is None or answer.strip() == "":
                print(f"Warning: Empty answer detected for sample {sample['id']}, using fallback direct inference...")
                
                fallback_messages = [
                    {"role": "user", "content": image_content + [{"type": "text", "text": sample['assistant_prompt'] + sample['question']}]}
                ]
                
                fallback_response = generate_qwen_response(model, processor, fallback_messages)
                print(f"Fallback Response: {fallback_response}")
                
                fallback_thinking, fallback_answer = parse_final_response(fallback_response)
                
                if fallback_answer and fallback_answer.strip() != "":
                    answer = fallback_answer
                    final_thinking = fallback_thinking if fallback_thinking else f"Fallback inference: {fallback_response}"
                    print(f"Fallback answer extracted: {answer}")
                else:
                    answer = fallback_response[:100] if fallback_response else "No response"
                    final_thinking = fallback_response
                    print(f"Using raw response as fallback answer: {answer}")
                
                with open(os.path.join(output_dir, "fallback.log"), "a") as log:
                    log.write(f"Sample {sample['id']}: Used fallback inference, answer: {answer}\n")

            output_dict = {
                "id": sample["id"],
                "original_id": sample.get("original_id", sample["id"]),
                "source_dataset": sample.get("source_dataset", "unknown"),
                "question_type": sample.get("question_type", "unknown"),
                "input_modality": sample.get("input_modality", "image"),
                "category": sample.get("category", "unknown"),
                "task": sample.get("task", "unknown"),
                "sub_task": sample.get("sub_task", "unknown"),
                "assistant_prompt": sample["assistant_prompt"],
                "question": sample["question"],
                "gt": sample["answer"],
                "response": answer,
                "final_thinking": final_thinking,
                "plan": tool_plan,
                "cot": cot_steps,
                "image_paths": sample["image_paths"],
            }

            # Unified result dict (for all_results.json - matching test_all_in_one.py format)
            result_entry = {
                "id": sample.get('id', idx),
                "source_dataset": sample.get("source_dataset", sample.get("source", "unknown")),
                "category": sample.get("category", "unknown"),
                "task": sample.get("task", "unknown"),
                "sub_task": sample.get("sub_task", "unknown"),
                "input_modality": sample.get("input_modality", "single-image"),
                "question_type": sample.get("question_type", "unknown"),
                "assistant_prompt": sample["assistant_prompt"],
                "question": sample.get("question", ""),
                "options": sample.get("options", []),
                "extra_info": sample.get("extra_info", {}),
                "img_paths": sample.get("img_paths", sample.get("image_paths", [])),
                "video_path": sample.get("video_path", None),
                "gt_answer": sample.get("answer", ""),
                "pred_answer": answer,
            }

            save_path = os.path.join(full_result_path, str(sample['id']))
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "response.json"), "w", encoding='utf-8') as f:
                json.dump(output_dict, f, indent=4, ensure_ascii=False)

            wf.write(json.dumps(output_dict) + "\n")
            wf.flush()

            # Append to all_results list
            all_results.append(result_entry)
            
            # Save all_results.json periodically (every 50 samples)
            if (idx - args.start + 1) % 20 == 0:
                with open(all_results_file, "w", encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"Saved intermediate results: {len(all_results)} items")

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            traceback.print_exc()
            with open(os.path.join(output_dir, "error.log"), "a") as log:
                log.write(f"Error processing example {idx}: {e}\n")

    # Final save of all_results.json
    with open(all_results_file, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "model_name": args.model_name,
        "dataset_path": args.data_path,
        "total_processed": len(all_results),
        "start_idx": args.start,
        "end_idx": args.end,
        "prompt_format": args.prompt_format
    }
    
    with open(os.path.join(output_dir, f"{run_id}_summary.json"), "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    wf.close()
    print(f"Processing completed: {len(all_results)} items processed")
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=0  python inference_plan-execute.py --start 0 --end 1000 --prompt_format cota