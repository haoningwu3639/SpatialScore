import os
import re
import torch
import base64
import matplotlib
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"
    
def process_depth_to_rgb(depth_image):
    # Convert to numpy array
    depth_array = np.array(depth_image, dtype=np.float32)
    
    # Normalize the depth values to 0-1 range
    if depth_array.max() > depth_array.min():
        normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
    else:
        normalized = np.zeros_like(depth_array)
    # Apply the Spectral_r colormap (similar to the reference code)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
    rgb_image = Image.fromarray(colored, mode="RGB")
    
    return rgb_image

# For InternVL, tool functions
def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return transform

# For InternVL, tool functions
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    area_threshold = 0.5 * image_size * image_size

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > area_threshold * ratio[0] * ratio[1]:
            best_ratio = ratio

    return best_ratio

# For InternVL, tool functions
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate possible aspect ratios within constraints
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width, target_height = image_size * target_aspect_ratio[0], image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    # Add thumbnail if needed
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

# For InternVL, tool functions
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images]).to(torch.float16)

    return pixel_values

# Result Parsing Functions
def extract_number(text: str) -> str:
    """Extract a number from text, handling digits and word representations.
    
    Args:
        text: Input text that may contain a number
        
    Returns:
        Extracted number as string or original cleaned text
    """
    # Handle direct number responses (common in the dataset)
    text = text.strip()
    if text.isdigit():
        return text
    
    # Look for the ASSISTANT: pattern at the end of longer responses
    assistant_match = re.search(r'ASSISTANT:\s*\(?([A-Fa-f])\)?', text)
    if assistant_match:
        return assistant_match.group(1).upper()
    
    # Normalize and clean the text
    clean_text = re.sub(r'</?(?:CONCLUSION|conclusion|ANSWER|answer|ASSISTANT|assistant)>', '', text).strip()
    text_lower = clean_text.lower()
    
    # First check for conclusion tags with numbers
    conclusion_match = re.search(r'<conclusion>\s*(\d+)\s*</conclusion>', text, re.IGNORECASE)
    if conclusion_match:
        return conclusion_match.group(1)
    
    # Dictionary for word-to-digit conversion - expanded
    word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
        'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10'
    }
    
    # Try to find numeric digits
    match = re.search(r'\d+(?:\.\d+)?', text_lower)
    if match:
        return match.group(0)
    
    # Look for number words in common phrases
    for word, digit in word_to_digit.items():
        if (text_lower.startswith(word) or 
            f" {word} " in text_lower or 
            f"there are {word}" in text_lower or 
            f"there is {word}" in text_lower or
            f"shows {word}" in text_lower or
            text_lower.endswith(f" {word}")):
            return digit
    
    # Try to find any number word in the text
    for word, digit in word_to_digit.items():
        if word in text_lower.split():
            return digit
    
    # Return original cleaned text if no number found
    return clean_text

# Result Parsing Functions
def extract_yes_no(text: str) -> str:
    """Extract yes/no response from text.
    
    Args:
        text: Input text containing a yes/no answer
        
    Returns:
        'Yes', 'No', or cleaned original text
    """
    # Handle direct yes/no responses
    text = text.strip()
    if text.lower() in ['yes', 'no']:
        return text.capitalize()
    
    # Look for the ASSISTANT: pattern at the end of longer responses
    assistant_match = re.search(r'ASSISTANT:\s*\(?([A-Fa-f])\)?', text)
    if assistant_match:
        return assistant_match.group(1).upper()
    
    # Normalize and clean the text
    clean_text = re.sub(r'</?(?:CONCLUSION|conclusion|ANSWER|answer|ASSISTANT|assistant)>', '', text).strip()
    text_lower = clean_text.lower()
    
    # Check for conclusion tags with yes/no
    conclusion_match = re.search(r'<conclusion>\s*(yes|no)\s*</conclusion>', text, re.IGNORECASE)
    if conclusion_match:
        return conclusion_match.group(1).capitalize()
    
    # Check for yes/no keywords with word boundary checks
    yes_patterns = [r'\byes\b', r'\byeah\b', r'\byep\b', r'\bcorrect\b', 
                   r'\btrue\b', r'\bright\b', r'\bagreed?\b']
    no_patterns = [r'\bno\b', r'\bnope\b', r'\bnot\b', r'\bfalse\b', 
                  r'\bwrong\b', r'\bincorrect\b', r'\bdisagreed?\b']
    
    for pattern in yes_patterns:
        if re.search(pattern, text_lower):
            return 'Yes'
            
    for pattern in no_patterns:
        if re.search(pattern, text_lower):
            return 'No'
    
    return clean_text

# Result Parsing Functions
def extract_option(text: str) -> str:
    """Extract a multiple-choice option (A-F) from text."""
    # Handle direct option responses
    text = text.strip()
    if re.match(r'^[A-Fa-f]\.?$', text):
        return text[0].upper()
    
    # Look for the ASSISTANT: pattern at the end of longer responses
    assistant_match = re.search(r'ASSISTANT:\s*\(?([A-Fa-f])\)?', text)
    if assistant_match:
        return assistant_match.group(1).upper()

    # Normalize text
    clean_text = re.sub(r'</?(?:CONCLUSION|conclusion|ANSWER|answer|ASSISTANT|assistant)>', '', text).strip()
    
    # Enhanced pattern for conclusion tags - captures (B) Yes. format
    conclusion_patterns = [
        # Match (Letter) followed by text
        r'<conclusion>\s*\(([A-Fa-f])\).*?</conclusion>',
        # Match regular letter formats in conclusion
        r'<conclusion>\s*\(?([A-Fa-f])\)?\.?\s*</conclusion>'
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Comprehensive patterns to match option formats
    patterns = [
        # Match answer or option labels
        r'(?i)(?:answer|option)[:\s]*\(?([A-Fa-f])\)?\.?',
        
        # Match letter in parentheses
        r'(?i)(?:^|\s)\(?([A-Fa-f])\)\.?(?:$|\s|\.)',
        
        # Match standalone letter with optional period
        r'(?i)(?:^|\s)([A-Fa-f])\.?(?:$|\s|\.|,)',
        
        # Match letters with periods
        r'(?i)(?:^|\s)([A-Fa-f])\.(?:$|\s)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    
    # If no structured format found, look for any option letter
    basic_match = re.search(r'[A-Fa-f]', text)
    if basic_match:
        return basic_match.group(0).upper()
    
    return clean_text

# Result Parsing Functions
def extract_numeric_with_unit(pred, gt=None, tolerance=2.0):
    """Extract numeric value with unit from predicted text and compare with ground truth if provided.
    
    Args:
        pred (str): Input text that may contain a numeric value with unit
        gt (str, optional): Ground truth string in format "value unit" (e.g., "5 meters")
        tolerance (float, optional): Ratio tolerance for comparison
        
    Returns:
        dict: Contains extracted value, unit, and comparison result
    """
    # Initialize return dictionary
    result = {"value": None, "unit": None, "is_correct": False}
    
    # Clean prediction text of special formatting
    pred = re.sub(r'</?(?:CONCLUSION|conclusion|ANSWER|answer|ASSISTANT|assistant)>', '', pred).strip()
    
    # STEP 1: Extract value with multiple patterns
    # Try formal LaTeX style format first
    value_match = re.search(r'\\scalar\{([^}]+)\}', pred)
    
    # Try markdown bold format (**2.828 meters**)
    if not value_match:
        value_match = re.search(r'\*\*([\d.]+)\s*(?:[a-zA-Z]+)\*\*', pred)
    
    # Try final value statement format (approximately **2.828 meters**)
    if not value_match:
        value_match = re.search(r'(?:approximately|about|roughly)\s*\**([\d.]+)\s*(?:[a-zA-Z]+)\**', pred, re.IGNORECASE)
    
    # Try standard formats like "5 meters" or "3.2 cm"
    if not value_match:
        value_match = re.search(r'(\d+\.?\d*)\s*(?:[a-zA-Z]+)', pred)
    
    # STEP 2: Extract unit with multiple patterns
    # Try formal LaTeX style format first
    unit_match = re.search(r'\\distance_unit\{([^}]+)\}', pred)
    
    # Try markdown bold format (**2.828 meters**)
    if not unit_match:
        unit_match = re.search(r'\*\*\d+\.?\d*\s*([a-zA-Z]+)\*\*', pred)
    
    # Try final value statement format (approximately **2.828 meters**)
    if not unit_match:
        unit_match = re.search(r'(?:approximately|about|roughly)\s*\**\d+\.?\d*\s*([a-zA-Z]+)\**', pred, re.IGNORECASE)
    
    # Try to find unit after a number
    if not unit_match:
        number_unit_match = re.search(r'\d+\.?\d*\s*([a-zA-Z]+)', pred)
        if number_unit_match:
            unit_match = number_unit_match

    # Process extracted value and unit
    if value_match:
        try:
            # Get value from the first matching group
            result["value"] = float(re.findall(r'\d+\.?\d*', value_match.group(1))[0])
        except (IndexError, ValueError):
            pass

    if unit_match:
        try:
            result["unit"] = unit_match.group(1).lower().strip()
        except (IndexError, AttributeError):
            pass

    # If ground truth provided, parse it and compare with extracted value
    if gt is not None and result["value"] is not None and result["unit"] is not None:
        # Parse ground truth value and unit from the combined string
        gt_match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z]+)', gt)
        if gt_match:
            try:
                gt_value = float(gt_match.group(1))
                gt_unit = gt_match.group(2).lower().strip()
                
                # Handle zero values safely
                if result["value"] == 0 or gt_value == 0:
                    # If both are zero, it's correct; if only one is zero, it's wrong
                    result["is_correct"] = (result["value"] == 0 and gt_value == 0)
                    return result
                    
                # Convert both values to centimeters for comparison
                multipliers = {
                    "m": 100, "meter": 100, "meters": 100, "metre": 100, "metres": 100,
                    "cm": 1, "centimeter": 1, "centimeters": 1, "mm": 0.1,
                    "ft": 30.48, "foot": 30.48, "feet": 30.48, "in": 2.54, "inch": 2.54, "inches": 2.54
                }
                
                # Get multipliers, defaulting to 1 if unit not found
                pred_multiplier = multipliers.get(result["unit"], 1)
                gt_multiplier = multipliers.get(gt_unit, 1)
                
                # Convert to common unit (cm)
                pred_value_cm = result["value"] * pred_multiplier
                gt_value_cm = gt_value * gt_multiplier
                
                # Check if values are within tolerance ratio
                try:
                    ratio = max(pred_value_cm / gt_value_cm, gt_value_cm / pred_value_cm)
                    result["is_correct"] = ratio < tolerance
                except ZeroDivisionError:
                    # This shouldn't happen with the earlier check, but just in case
                    result["is_correct"] = False
            except (IndexError, ValueError):
                pass
    
    return result

# Tool function for VSI-Bench
def abs_dist_norm(pred: float, target: float) -> float:
    """For VSI-Bench, calculate normalized absolute distance (relative error)."""
    return abs(pred - target) / target if target != 0 else float('inf')

# Tool function for VSI-Bench
def mean_relative_accuracy(pred: float, target: float, start: float, end: float, interval: float) -> float:
    """For VSI-Bench, calculate Mean Relative Accuracy for open-ended questions."""
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    
    return accuracy.mean()