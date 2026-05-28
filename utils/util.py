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
    """Extract a number from text, handling digits and number words.
    In counting scenarios, the last numeric occurrence is considered final.

    Args:
        text: input text that may contain one or more numbers.

    Returns:
        Extracted number string or cleaned text.
    """
    import re
    # Remove assistant markup etc.
    text = re.sub(r'</?(?:CONCLUSION|conclusion|ANSWER|answer|ASSISTANT|assistant|think|/think)>', '', text)
    text = text.strip()

    # Direct numeric
    if text.isdigit():
        return text

    # Extract all numeric occurrences
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        # Return the last one (most likely the final answer)
        return numbers[-1]

    # Handle number words (one, two, three...)
    word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }

    # Convert text to lowercase for matching
    text_lower = text.lower()

    # Match any word number, choose last occurrence
    found_words = [word_to_digit[w] for w in word_to_digit if w in text_lower.split()]
    if found_words:
        return found_words[-1]

    # Default: return cleaned text if no numeric info found
    return text.strip()


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

# Add this helper inside your util (above or inside extract_numeric_with_unit)
def _collapse_ranges_to_max(text: str) -> str:
    """
    Replace numeric ranges like '10-15 cm', '10 ~ 15cm', '10 to 15 meters'
    with their upper bound: '15 cm' / '15 cm' / '15 meters'.
    If no unit is captured, we only replace numbers; unit can be borrowed from GT later.
    """
    import re
    # Pattern captures: left value, right value, optional unit after the right value
    rng = re.compile(r'(-?\d+(?:\.\d+)?)\s*(?:-|–|—|~|to)\s*(-?\d+(?:\.\d+)?)(?:\s*([A-Za-z]+))?', re.IGNORECASE)
    # Iterate until no more ranges remain (handles multiple ranges in one string)
    while True:
        m = rng.search(text)
        if not m:
            break
        try:
            v1 = float(m.group(1)); v2 = float(m.group(2))
            vmax = max(v1, v2)
            unit = m.group(3) or ""   # may be None; keep empty to allow later unit inference
            replacement = f"{vmax} {unit}".strip()
            text = text[:m.start()] + replacement + text[m.end():]
        except Exception:
            # On any parsing issue, break to avoid infinite loop
            break
    return text


# Result Parsing Functions
def extract_numeric_with_unit(pred, gt=None, tolerance=2.0):
    import re

    # ---- 特殊处理：明确拒绝无法测量距离的回答 ----
    if isinstance(pred, str):
        if "unable" in pred.lower() and "distance" in pred.lower():
            return {
                "value": None, "unit": None, "is_correct": False,
                "answer_value": None, "answer_unit": None,
                "gt_value": None, "gt_unit": None
            }

    # ---- Canonical unit mapping and cm multipliers ----
    unit_alias = {
        "m": "meter", "meter": "meter", "metre": "meter", "meters": "meter", "metres": "meter",
        "cm": "centimeter", "centimeter": "centimeter", "centimeters": "centimeter",
        "mm": "millimeter", "millimeter": "millimeter", "millimeters": "millimeter",
        "in": "inch", "inch": "inch", "inches": "inch",
        "ft": "foot", "feet": "foot", "foot": "foot",
    }
    multipliers = {
        "meter": 100,
        "centimeter": 1,
        "millimeter": 0.1,
        "inch": 2.54,
        "foot": 30.48,
        "m": 100, "cm": 1, "mm": 0.1, "in": 2.54, "ft": 30.48,
    }

    def canonical_unit(u: str | None) -> str | None:
        if not u:
            return None
        u = u.lower().strip()
        return unit_alias.get(u, u)

    # ---- Initialize result ----
    result = {
        "value": None, "unit": None, "is_correct": False,
        "answer_value": None, "answer_unit": None,
        "gt_value": None, "gt_unit": None
    }

    # ---- Clean general wrappers ----
    text = str(pred)
    text = re.sub(r'</?(?:CONCLUSION|conclusion|ANSWER|answer|ASSISTANT|assistant)>', '', text).strip()

    # ---- Ignore think block: only keep the text AFTER "◁/think▷" if present ----
    end_think_pos = text.rfind("◁/think▷")
    if end_think_pos != -1:
        text_to_parse = text[end_think_pos + len("◁/think▷"):]
    else:
        text_to_parse = text

    text_to_parse = _collapse_ranges_to_max(text_to_parse)

    # ---- Parse GT ----
    gt_value_cm, gt_unit_std = None, None
    if gt is not None:
        gt = str(gt)
        m_gt = re.search(r'(-?\d+\.?\d*)\s*([A-Za-z]+)', gt)
        if m_gt:
            try:
                gt_v = float(m_gt.group(1))
                gt_u = canonical_unit(m_gt.group(2))
                mul = multipliers.get(gt_u, 1)
                gt_value_cm = gt_v * mul
                gt_unit_std = gt_u if gt_u in multipliers else "centimeter"
            except Exception:
                pass

    # ---- Collect candidates ----
    candidates: list[tuple[float, str]] = []

    scalars = [m.group(1) for m in re.finditer(r'\\scalar\{([^}]+)\}', text_to_parse)]
    d_units = [m.group(1) for m in re.finditer(r'\\distance_unit\{([^}]+)\}', text_to_parse)]
    if scalars:
        try:
            val = float(re.findall(r'-?\d+\.?\d*', scalars[-1])[0])
            if d_units:
                u = canonical_unit(d_units[-1])
                if u:
                    candidates.append((val, u))
        except Exception:
            pass

    for m in re.finditer(r'(-?\d+(?:\.\d+)?)\s*\\\s*([A-Za-z]+)', text_to_parse):
        val = float(m.group(1))
        u = canonical_unit(m.group(2))
        if u:
            candidates.append((val, u))

    for m in re.finditer(r'\*\*\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*\*\*', text_to_parse):
        val = float(m.group(1))
        u = canonical_unit(m.group(2))
        if u:
            candidates.append((val, u))

    unit_token = r'(meters?|metres?|centimeters?|millimeters?|cm|mm|m|inches?|inch|feet|foot|ft)'
    for m in re.finditer(rf'(-?\d+(?:\.\d+)?)\s*{unit_token}', text_to_parse, flags=re.I):
        val = float(m.group(1))
        u = canonical_unit(m.group(2))
        if u:
            candidates.append((val, u))

    parsed_value, parsed_unit = None, None
    if candidates:
        parsed_value, parsed_unit = candidates[-1]
    else:
        # ---- Fallback: number only (borrow unit from GT if available) ----
        m_num = re.findall(r'-?\d+(?:\.\d+)?', text_to_parse)
        if m_num:
            try:
                parsed_value = float(m_num[-1])
                if parsed_value <= 0 or parsed_value > 1e4:
                    parsed_value = None
                    parsed_unit = None
                else:
                    parsed_unit = gt_unit_std if gt_unit_std else None
            except Exception:
                parsed_value = None
                parsed_unit = None

    result["value"] = parsed_value
    result["unit"] = parsed_unit

    if parsed_value is None or parsed_unit is None:
        return result

    if gt_value_cm is not None and parsed_value is not None and parsed_unit:
        pred_cm = parsed_value * multipliers.get(parsed_unit, 1)
        result["answer_value"] = pred_cm
        result["answer_unit"] = "centimeter"
        result["gt_value"] = gt_value_cm
        result["gt_unit"] = "centimeter"

        if pred_cm == 0 or gt_value_cm == 0:
            result["is_correct"] = (pred_cm == gt_value_cm)
        else:
            try:
                ratio = max(pred_cm / gt_value_cm, gt_value_cm / pred_cm)
                result["is_correct"] = (ratio < tolerance)
            except ZeroDivisionError:
                result["is_correct"] = False

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


def main():

    result=extract_numeric_with_unit("I'm unable to measure precise distances in 3D space from 2D images.","2.5 centimeter")
    print(result)

if __name__ == "__main__":
    main()
