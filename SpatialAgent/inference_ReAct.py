import os
import re
import json
import torch
import argparse
import traceback
from qwen_vl_utils import process_vision_info

from autogen.agentchat import AssistantAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

from utils.parser import Parser
from utils.executor import Executor
from utils.prompt import DirectAnswerPrompt, CoTAPrompt, FeedbackPrompt

from agent import UserAgent
from data import AgentDataset
from multimodal_chat_agent import MultimodalChatAgent
from action import (
                    SelfThinking ,Detect3DObjects,
                    LocalizeObjects, CountObjects, GetObjectMask, EstimateRegionDepth, 
                    EstimateObjectDepth, GetCameraParametersVGGT, EstimateObjectGeometryProperties, 
                    MatchImagesSIFT, EstimateHomographyMatrix, EstimateOpticalFlow, GetObjectOrientation, 
                    Terminate, EstimateImageDepth, Get3DDistance
                )

if "OPENAI_API_KEY" not in os.environ or os.environ.get("OPENAI_API_KEY", "").strip() == "":
    os.environ["OPENAI_API_KEY"] = "sk-dummy-key-to-prevent-validation-errors"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-dummy-key-to-prevent-validation-errors")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help="the random seed used in autogen.")
    parser.add_argument('--max-reply', default=10, type=int, help="the maximum number of replies after the first attempt.")
    parser.add_argument('--prompt_format', default='cota', type=str, help="the inference prompt format", choices=['cota', 'direct'])
    parser.add_argument('--data_path', type=str, default="../SpatialScore_benchmark/SpatialScore_benchmark.ndjson")
    parser.add_argument('--model_name', type=str, default='qwen3vl-4b', choices=['qwen2_5vl-7b', 'qwen2_5vl-32b', "qwen2_5vl-72b", 'qwen3vl-2b', 'qwen3vl-4b', "qwen3vl-8b", "qwen3vl-32b", 'gpt-4o-mini', 'gpt-4o-2024-08-06'])
    parser.add_argument('--model_path', type=str, default="../huggingface/Qwen3-VL-4B-Instruct")
    parser.add_argument('--output_dir', default='./output-ReAct', type=str, help="output directory.")
    parser.add_argument('--exp-id', default="", type=str, help="a unique string for identifying the current experiment.")
    parser.add_argument('--execute', action='store_true', help="whether to use feedback from execution or not.")
    parser.add_argument('--language-only', action='store_true', help="whether to use language only model or not (only for GPT models).")
    parser.add_argument('--simple-prompt', action='store_true', help="whether to use simplified prompt (without action descriptions and examples) or not.")

    parser.add_argument('--start', type=int, default=0, help="start index of dataset to process")
    parser.add_argument('--end', type=int, default=-1, help="end index of dataset to process (inclusive)")

    args = parser.parse_args()
    return args


def create_action_registry(model=None, processor=None, depth_estimator=None, late_load=True):
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

def format_query(image_paths, query):
    if isinstance(image_paths, list):
        prompt = ""
        for i, image_path in enumerate(image_paths):
            prompt += f"Image {i}: <img {image_path}>\n"
        prompt += f"Question: {query}"
        return prompt
    else:
        raise ValueError("image_paths should be a list of image paths.")

def checks_terminate_message(msg):
    if isinstance(msg, str):
        return msg.find("Terminate") > -1
    elif isinstance(msg, dict) and 'content' in msg:
        return msg['content'].find("Terminate") > -1
    else:
        raise NotImplementedError
    
def extract_answer_from_text(text):
    """Extract answer from model response text"""
    if not text or not isinstance(text, str):
        return None
    
    # Try to find answer in tags
    answer_match = re.search(r'<answer>(.*?)(?:</answer>|$)', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Try to find in quotes
    quote_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if quote_match:
        return quote_match.group(1).strip()
    
    # Return first 100 chars as fallback
    return text[:100] if text else None

def generate_direct_inference(mm_agent, sample):
    NUM_BEAMS, TEMPERATURE, MAX_NEW_TOKENS, USE_CACHE = 1, 0.0, 1024, True
    DO_SAMPLE = True if TEMPERATURE > 0 else False
    device = mm_agent.model.device
    
    image_paths = sample['image_paths']
    
    image_content = [{"type": "image", "image": img_path} for img_path in image_paths]
    messages = [
        {"role": "user", "content": image_content + [{"type": "text", "text": sample['assistant_prompt'] + sample['question']}]}
    ]
    
    input_text = mm_agent.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = mm_agent.processor(
        text=[input_text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        output_ids = mm_agent.model.generate(
            **inputs, 
            num_beams=NUM_BEAMS, 
            temperature=TEMPERATURE, 
            max_new_tokens=MAX_NEW_TOKENS, 
            use_cache=USE_CACHE, 
            do_sample=DO_SAMPLE
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        response = mm_agent.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    
    torch.cuda.empty_cache()
    
    return response

def main():
    args = get_args()

    run_id = f"{args.exp_id if args.exp_id else ''}-{args.prompt_format}-max-reply-{str(args.max_reply)}-seed-{str(args.seed)}-{str(args.start)}-{str(args.end)}".replace('/', '-')
    # run_id = f"{args.exp_id if args.exp_id else ''}-cota-max-reply-{str(args.max_reply)}-seed-{str(args.seed)}-{str(args.start)}-{str(args.end)}".replace('/', '-')
    model_name = args.model_name.replace('/', '-')
    output_dir = os.path.join(args.output_dir, model_name)

    full_input_path = os.path.join(args.data_path)
    full_result_path = os.path.join(args.output_dir, model_name, run_id)
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

    # Create actions with shared tool instances
    depth_estimator = EstimateImageDepth()
    action_registry = create_action_registry(model=None, processor=None, depth_estimator=depth_estimator, late_load=False if args.model_name in ["gpt-4o-2024-08-06", "gpt-4o-mini"] else True)
    print("Successful Tool Initialization.")

    # Set up executor, prompt generator, parser, and user agent
    executor = Executor(input_folder=full_input_path, result_folder=full_result_path, action_registry=action_registry) if args.execute else None 

    actions = list(action_registry.values())

    if args.prompt_format == 'cota':
        prompt_generator = CoTAPrompt(actions=actions)
    elif args.prompt_format == 'direct':
        prompt_generator = DirectAnswerPrompt(actions=[])
    
    parser = Parser(prompt_generator=prompt_generator) 
    feedback_generator = FeedbackPrompt()
    user = UserAgent(
        name="user_agent",
        human_input_mode='NEVER',
        max_consecutive_auto_reply=args.max_reply,
        is_termination_msg=checks_terminate_message,
        prompt_generator=prompt_generator,
        feedback_generator=feedback_generator,
        parser=parser,
        executor=executor,
        code_execution_config={'use_docker': False}
    )

    if args.simple_prompt:
        all_action_names = [action.name for action in actions]
        all_actions_str = ", ".join(all_action_names)
        system_prompt = f"""[BEGIN OF GOAL] You are a helpful assistant, and your goal is to solve the # USER REQUEST #. You can either rely on your own capabilities or perform actions with external tools to help you. You can use these actions: {all_actions_str} [END OF GOAL]"""
    else:
        system_prompt = prompt_generator.get_task_prompt_only()
    print(system_prompt)

    data = AgentDataset(args.data_path)
    print("Successful Data Initialization")

    if args.end == -1:
        args.end = len(data) - 1
    # Initialize agents 
    all_messages = {}
    if args.model_name in ["gpt-4o-2024-08-06", "gpt-4o-mini"]:
        config_list = [{"model": args.model_name, "api_key": OPENAI_API_KEY, "base_url": "https://chatapi.zjt66.top/v1"}]
        llm_config = {"seed": args.seed, "config_list": config_list, "cache_seed": None}
        if args.language_only:
            mm_agent = AssistantAgent(name="planner", llm_config=llm_config, human_input_mode='NEVER', system_message=system_prompt)
        else:
            mm_agent = MultimodalConversableAgent(name="mm_agent", system_message=system_prompt, human_input_mode='NEVER', max_consecutive_auto_reply=args.max_reply, llm_config=llm_config)

    elif args.model_name in ['qwen3vl-2b', 'qwen3vl-4b', "qwen3vl-8b", "qwen3vl-32b", "qwen2_5vl-7b", "qwen2_5vl-32b", "qwen2_5vl-72b"]:
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "sk-dummy"
        
        config_list = [{"model": args.model_name, "model_path": args.model_path}]
        llm_config = {"seed": args.seed, "config_list": config_list}
        mm_agent = MultimodalChatAgent(name="multimodal_chat_agent", system_message=system_prompt, human_input_mode='NEVER', max_consecutive_auto_reply=args.max_reply, llm_config=llm_config)
        action_registry["SelfThinking"].model, action_registry["SelfThinking"].processor = mm_agent.model, mm_agent.processor

    else:
        raise ValueError(f"Unsupported model: {args.model_name}.")

    # Run experiment on all examples
    for idx in range(args.start, args.end + 1):
        sample = data[idx]
        
        if 'id' in sample and sample['id'] in has_inferenced:
            print(f"Skipping already processed sample with ID: {sample['id']}")
            continue
            
        print(f"Processing sample with ID: {sample['id']}")
        print(sample)

        prediction = None
        failed = False
        all_messages = {}
            
        image_paths, query = sample['image_paths'], sample['assistant_prompt'] + sample['question']
        formatted_query = format_query(image_paths=image_paths, query=query)
        
        try:
            user.initiate_chat(assistant=mm_agent, message=formatted_query, task=sample)
            all_messages = mm_agent.chat_messages
            prediction = user.final_answer
            failed = False

        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")
            traceback.print_exc()
            failed = True
            with open(os.path.join(output_dir, "error.log"), "a") as log:
                log.write(f"Error processing example {idx}: {e}\n")

        # Fallback mechanism for empty predictions
        if not prediction or prediction is None or (isinstance(prediction, str) and prediction.strip() == ""):
            print(f"Warning: Empty prediction for sample {sample['id']}, attempting fallback inference...")
            
            try:
                if all_messages and 'user_agent' in all_messages:
                    msg = all_messages['user_agent'][-1]
                    if isinstance(msg, dict) and 'content' in msg:
                        fallback_text = msg['content']
                        if isinstance(fallback_text, list):
                            for content in fallback_text:
                                if isinstance(content, dict) and 'text' in content:
                                    extracted = extract_answer_from_text(content['text'])
                                    if extracted:
                                        prediction = extracted
                                        break
                        elif isinstance(fallback_text, str):
                            extracted = extract_answer_from_text(fallback_text)
                            if extracted:
                                prediction = extracted
                
                if not prediction or (isinstance(prediction, str) and prediction.strip() == ""):
                    print(f"Direct message extraction failed for sample {sample['id']}, calling model for direct inference...")
                    
                    direct_response = generate_direct_inference(mm_agent, sample)
                    print(f"Direct inference response: {direct_response}")
                    
                    extracted = extract_answer_from_text(direct_response)
                    if extracted and extracted.strip() != "":
                        prediction = extracted
                    else:
                        prediction = direct_response[:100] if direct_response else "Failed"
                    
                    print(f"Fallback direct inference answer: {prediction}")
                
                with open(os.path.join(output_dir, "fallback.log"), "a") as log:
                    log.write(f"Sample {sample['id']}: Used fallback inference, prediction: {prediction}\n")
                    
            except Exception as fallback_e:
                print(f"Fallback inference also failed for sample {sample['id']}: {fallback_e}")
                traceback.print_exc()
                prediction = "Fallback failed"
                with open(os.path.join(output_dir, "fallback_error.log"), "a") as log:
                    log.write(f"Fallback failed for sample {sample['id']}: {fallback_e}\n")

            except Exception as fallback_e:
                print(f"Fallback inference also failed for sample {sample['id']}: {fallback_e}")
                prediction = "Fallback failed"
                with open(os.path.join(output_dir, "fallback_error.log"), "a") as log:
                    log.write(f"Fallback failed for sample {sample['id']}: {fallback_e}\n")

        if args.output_dir and not failed:
            messages = {agent.name: msg for agent, msg in all_messages.items()}
            sample['all_messages'] = messages
            sample['prediction'] = user.final_answer
            sample['called_tools'] = user.called_tools
            sample['image_paths'] += user.new_image_paths
            
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
                "response": prediction,
                "called_tools": user.called_tools if hasattr(user, 'called_tools') else None,
                "image_paths": sample["image_paths"],
            }
            
            # Save chat messages separately as pickle file
            all_msgs = {agent.name: msg for agent, msg in all_messages.items()}
            save_path = os.path.join(full_result_path, str(sample['id']))
            os.makedirs(save_path, exist_ok=True)

            print("Output Dict:", output_dict)

            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif hasattr(obj, '__class__') and obj.__class__.__module__ != 'builtins':
                    # For non-builtin objects (like PIL.Image), convert to a string representation
                    return str(obj)
                else:
                    return obj

            all_msg_path = os.path.join(save_path, "all_messages.json")
            with open(all_msg_path, "w", encoding='utf-8') as f:
                json.dump(make_json_serializable(all_msgs), f, indent=4, ensure_ascii=False)
            
            output_dict["all_messages"] = all_msg_path

            wf.write(json.dumps(output_dict) + "\n")
            wf.flush()

            # **Unified result dict (for all_results.json - matching test_all_in_one.py format)**
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
                "pred_answer": prediction,
            }
            
            # Append to all_results list
            all_results.append(result_entry)
            
            # Save all_results.json periodically (every 20 samples)
            if (idx - args.start + 1) % 20 == 0:
                with open(all_results_file, "w", encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"Saved intermediate results: {len(all_results)} items")

        user.reset()
        mm_agent.reset()
    
    # **Final save of all_results.json**
    with open(all_results_file, "w", encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # **Save summary**
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

# CUDA_VISIBLE_DEVICES=0  python inference_ReAct.py --start 0 --end 1000 --execute --prompt_format cota --model_path /remote-home/share/huggingface/Qwen3-VL-4B-Instruct --model_name qwen3vl-4b
