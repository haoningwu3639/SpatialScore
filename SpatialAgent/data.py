import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def resolve_image_path(image_path, dataset_json_path):
    """
    Resolve image path relative to dataset directory.
    Tries multiple strategies to find the file.
    """
    # If already absolute path and exists, use it
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    
    # Strategy 1: Relative to dataset directory
    dataset_dir = os.path.dirname(os.path.abspath(dataset_json_path))
    resolved_path = os.path.join(dataset_dir, image_path)
    if os.path.exists(resolved_path):
        return resolved_path
    
    # Strategy 2: Relative to parent of dataset directory (common structure)
    parent_dir = os.path.dirname(dataset_dir)
    resolved_path = os.path.join(parent_dir, image_path)
    if os.path.exists(resolved_path):
        return resolved_path
    
    # Strategy 3: Relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_path = os.path.join(script_dir, "..", image_path)
    resolved_path = os.path.abspath(resolved_path)
    if os.path.exists(resolved_path):
        return resolved_path
    
    # Strategy 4: Try as-is (might be absolute or relative to cwd)
    if os.path.exists(image_path):
        return os.path.abspath(image_path)
    
    # If nothing worked, return the best guess (Strategy 1) anyway
    # This allows error messages to show the expected path
    return os.path.join(dataset_dir, image_path)


class AgentDataset(Dataset):
    def __init__(self, dataset_json_path, model_name="qwen2_5vl-7b"):
        self.dataset_json_path = os.path.abspath(dataset_json_path)
        self.model_name = model_name
        self.data = self._load_and_process_data()

    def _load_and_process_data(self):
        """Load NDJSON data and process multi-choice questions"""
        data = []
        with open(self.dataset_json_path, 'r', encoding='utf-8') as f:
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
                    
                    # **关键：直接在这里解析所有图像路径，替换原始路径**
                    resolved_img_paths = []
                    img_paths = item.get('img_paths', item.get('image_paths', []))
                    if isinstance(img_paths, str):
                        img_paths = [img_paths]
                    
                    for img_path in img_paths:
                        resolved_path = resolve_image_path(img_path, self.dataset_json_path)
                        resolved_img_paths.append(resolved_path)
                    
                    item['image_paths'] = resolved_img_paths
                    item['img_paths'] = resolved_img_paths
                    
                    data.append(item)
        
        return data
    
    def _process_model_input(self, sample):
        """Process model input based on question type and model name"""
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
        img_paths = sample.get('image_paths', [])
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        
        # Load images based on model type
        images = []
        for img_path in img_paths:
            if os.path.exists(img_path):
                images.append(Image.open(img_path).convert('RGB'))
            else:
                print(f"Warning: Image not found at {img_path}")
        
        # Update sample with processed data
        processed_sample = sample.copy()
        processed_sample['images'] = images
        processed_sample['assistant_prompt'] = assistant_prompt
        
        return processed_sample
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx].copy()
        processed_sample = self._process_model_input(sample)
        
        return processed_sample

    def get_metadata(self):
        """Get dataset metadata"""
        question_types, source_datasets, categories, input_modalities = {}, {}, {}, {}
        
        for item in self.data:
            # Count question types
            q_type = item.get('question_type', 'unknown')
            question_types[q_type] = question_types.get(q_type, 0) + 1
            
            # Count source datasets
            source = item.get('source_dataset', item.get('source', 'unknown'))
            source_datasets[source] = source_datasets.get(source, 0) + 1
            
            # Count categories
            category = item.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            # Count input modalities
            modality = item.get('input_modality', 'unknown')
            input_modalities[modality] = input_modalities.get(modality, 0) + 1
        
        return {
            'total_samples': len(self.data),
            'question_types': question_types,
            'source_datasets': source_datasets,
            'categories': categories,
            'input_modalities': input_modalities
        }
    
    def filter_by_question_type(self, question_types):
        """Create a filtered dataset by question type"""
        if isinstance(question_types, str):
            question_types = [question_types]
        
        filtered_data = [item for item in self.data if item.get('question_type') in question_types]
        
        # Create a new dataset instance with filtered data
        new_dataset = AgentDataset.__new__(AgentDataset)
        new_dataset.dataset_json_path = self.dataset_json_path
        new_dataset.model_name = self.model_name
        new_dataset.data = filtered_data
        
        return new_dataset
    
# Usage example and utility functions
def create_dataloader(dataset_json_path, model_name="qwen2_5vl-7b", batch_size=1, shuffle=False, num_workers=0):
    """Create a DataLoader for the AgentDataset"""
    from torch.utils.data import DataLoader
    
    dataset = AgentDataset(dataset_json_path, model_name)
    
    # Custom collate function to handle different data types
    def collate_fn(batch):
        return batch  # Return list of samples as-is since they contain mixed types
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    return dataloader, dataset


if __name__ == "__main__":
    # Create dataset
    dataset_path = "../SpatialScore_benchmark/SpatialScore_benchmark.ndjson"
    model_name = "qwen2_5vl-7b"
    
    dataset = AgentDataset(dataset_path, model_name)
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print("Dataset metadata:")
    metadata = dataset.get_metadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    # Create filtered datasets
    multi_choice_dataset = dataset.filter_by_question_type('multi-choice')
    print(f"\nMulti-choice dataset size: {len(multi_choice_dataset)}")
    
    # Create dataloader
    dataloader, _ = create_dataloader(dataset_path, model_name, batch_size=1)
    print(f"\nDataLoader created with {len(dataloader)} batches")

    for i, batch in enumerate(dataloader):
        print(batch)
        