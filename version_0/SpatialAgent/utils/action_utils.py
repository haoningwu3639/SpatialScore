import io
import base64
from PIL import Image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  
    
    return {
        "image_data": base64.b64encode(buffered.getvalue()).decode('utf-8'),
        "metadata": {"format": image.format, "size": image.size, "mode": image.mode}
    }

def base64_to_image(data):
    return Image.open(io.BytesIO(base64.b64decode(data["image_data"])))

def expand_bbox(bbox, original_image_size, margin=0.5):
    """
        Expands the bounding box by half of its width and height.
        bbox: A tuple (left, top, right, bottom)
        original_image_size: A tuple (width, height) of the original image size
        return: A tuple (new_left, new_top, new_right, new_bottom)
    """
    left, upper, right, lower = bbox
    width = right - left
    height = lower - upper

    # Calculate the new width and height
    new_width = width * (1 + margin) if margin <= 1.0 else width + margin
    new_height = height * (1 + margin) if margin <= 1.0 else height + margin

    # Calculate the center of the original bounding box
    center_x = left + width / 2
    center_y = upper + height / 2

    # Determine the new bounding box coordinates
    new_left = max(0, center_x - new_width / 2)
    new_upper = max(0, center_y - new_height / 2)
    new_right = min(original_image_size[0], center_x + new_width / 2)
    new_lower = min(original_image_size[1], center_y + new_height / 2)

    return (int(new_left), int(new_upper), int(new_right), int(new_lower))

