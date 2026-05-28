import torch
import logging
import requests
from PIL import Image
from typing import List, Optional, Tuple

from autogen.agentchat.agent import Agent
from autogen.code_utils import content_str
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class MultimodalChatAgent(MultimodalConversableAgent):
    def __init__(self, name: str, system_message: Optional[Tuple[str, List]] = "", is_termination_msg: str = None, *args, **kwargs):
        """
        Args:
            name (str): agent name.
            system_message (str): system message for the ChatCompletion inference. Please override this attribute if you want to reprogram the agent.
            **kwargs (dict): Please refer to other kwargs in
                [ConversableAgent](../conversable_agent#__init__).
        """
        super().__init__(name, system_message=system_message, is_termination_msg=is_termination_msg, *args, **kwargs)
        
        self.update_system_message(system_message)
        
        print(f"system message: {self.system_message}")
        
        self._is_termination_msg = (
            is_termination_msg
            if is_termination_msg is not None
            else (lambda x: content_str(x.get("content")) == "TERMINATE")
        )
        
        self.register_reply([Agent, None], reply_func=MultimodalChatAgent._image_reply, position=0)
        config = self.llm_config['config_list'][0]
        self.model_type = None

        if 'qwen' in config['model'].lower():
            self.model_type = 'qwen'
            self.model = AutoModelForImageTextToText.from_pretrained(config['model_path'], device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2")
            self.processor = AutoProcessor.from_pretrained(config['model_path'], use_fast=False, trust_remote_code=True, min_pixels=256*28*28, max_pixels=2560*28*28)
        
    def _image_reply(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        # The formats for Qwen and GPT are different, so we handle them here
        images, image_paths = [], []
        
        for msg in messages:
            if msg["role"] == "user":
                for item in msg["content"]:
                    if item["type"] == "image_url":
                        image_url = item["image_url"]["url"]
                        # Check if it's a string before using startswith
                        if isinstance(image_url, str):
                            if image_url.startswith("data:"):
                                # Handle base64 encoded images
                                image = Image.open(requests.get(image_url, stream=True).raw)
                                images.append(image)
                            else:
                                # Handle file path
                                image_paths.append(image_url)
                                img = Image.open(image_url).convert('RGB')
                                images.append(img)
                        elif isinstance(image_url, Image.Image):
                            # If it's already an Image object
                            images.append(image_url)
                            # We don't have a path for this image
                        else:
                            logger.warning(f"Unsupported image format: {type(image_url)}")
        
        # Generate response based on model type
        if self.model_type == 'qwen':
            response = self._generate_qwen_response(images, image_paths)
        else:
            error_msg = f"Unsupported model type: {self.model_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return True, response

    def _generate_qwen_response(self, images, image_paths):
        """Generate a response using Qwen2.5VL model."""
        NUM_BEAMS, TEMPERATURE, MAX_NEW_TOKENS, USE_CACHE = 1, 0.0, 4096, True
        DO_SAMPLE = True if TEMPERATURE > 0 else False
        
        device = next(self.model.parameters()).device
        
        # Retrieve full message history from self._oai_messages
        messages = []
        for sender in self._oai_messages:
            messages.extend(self._oai_messages[sender])
        
        # Sort messages by their inherent order to maintain conversation flow
        # (This assumes messages come with some timestamp or ordering index; if not, they're already in order)
        
        # Prepare Qwen-formatted messages
        qwen_messages = []
        
        # Add system message first
        qwen_messages.append({"role": "system", "content": content_str(self.system_message)})
        
        # Process all messages to build conversation history
        for msg in messages:
            if msg["role"] == "user":
                user_content = []
                
                # Extract text content
                for item in msg["content"]:
                    if item["type"] == "text":
                        user_content.append({"type": "text", "text": item["text"]})
                
                # Handle images (only for the latest message with images)
                has_images = any(item["type"] == "image_url" for item in msg["content"] if isinstance(item, dict))
                if has_images and (images or image_paths):
                    if image_paths:
                        for img_path in image_paths:
                            user_content.append({"type": "image", "image": img_path})
                    else:
                        for img in images:
                            user_content.append({"type": "image", "image": img})
                
                if user_content:  # Only add if there's actual content
                    qwen_messages.append({"role": "user", "content": user_content})
                    
            elif msg["role"] == "assistant":
                assistant_content = content_str(msg["content"])
                if assistant_content:
                    qwen_messages.append({"role": "assistant", "content": assistant_content})
        
        # Process text and images
        input_text = self.processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
        
        # Extract images from the message content
        image_inputs = []
        for msg in qwen_messages:
            if isinstance(msg.get("content"), list):
                for content in msg["content"]:
                    if content.get("type") == "image":
                        image_path = content["image"]
                        if isinstance(image_path, str):
                            try:
                                image = Image.open(image_path).convert("RGB")
                                image_inputs.append(image)
                            except Exception as e:
                                logger.warning(f"Failed to open image {image_path}: {e}")
                        else:
                            image_inputs.append(image_path)
        
        # Prepare inputs
        inputs = self.processor(text=[input_text], images=image_inputs if image_inputs else None, padding=True, return_tensors="pt").to(device)
        
        # Generate output
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                num_beams=NUM_BEAMS, 
                temperature=TEMPERATURE, 
                max_new_tokens=MAX_NEW_TOKENS, 
                use_cache=USE_CACHE, 
                do_sample=DO_SAMPLE
            )
            
            # Extract only the newly generated tokens
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip()
        
        # Clean up memory
        torch.cuda.empty_cache()
        
        return response