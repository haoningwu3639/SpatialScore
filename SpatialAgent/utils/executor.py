import os
from PIL import Image
from utils.observation import BaseObservation

class Executor:
    def __init__(self, input_folder: str, result_folder: str, action_registry=None) -> None:
        self.result_folder = result_folder
        self.input_folder = input_folder
        self.action_registry = action_registry or {}

    # def get_full_image_path(self, task, image_filename):
    #     input_image_id = int(image_filename.split("-")[1])
    #     num_input_images = len(task['image_paths'])
    #     if input_image_id < num_input_images: # this image argument is one of the input images
    #         full_path = task['image_paths'][input_image_id]
    #     else:
    #         full_path = os.path.join(self.full_result_path, f"{image_filename}.jpg")
    #     return full_path


    def get_full_image_path(self, task, image_filename):
        """
        Convert image filename(s) to full path(s) using task['image_paths'] or result folder.
        Supports single filename (str) or list of filenames (List[str]).
        """
        print(f"get_full_image_path called with image_filename={image_filename}, task_id={task['id']}")
        if isinstance(image_filename, list):
            # Handle list of filenames
            full_paths = []
            for fname in image_filename:
                full_paths.append(self._get_single_image_path(task, fname))
            print(f"Returning full paths for list: {full_paths}")
            return full_paths
        else:
            # Handle single filename
            full_path = self._get_single_image_path(task, image_filename)
            print(f"Returning full path for single: {full_path}")
            return full_path

    def _get_single_image_path(self, task, image_filename):
        """
        Helper method to process a single image filename.
        """
        print(f"_get_single_image_path called with image_filename={image_filename}")
        if not isinstance(image_filename, str):
            raise ValueError(f"Expected string for image_filename, got {type(image_filename)}: {image_filename}")

        # Check if filename is a placeholder like 'image-0'
        if image_filename.startswith("image-"):
            try:
                input_image_id = int(image_filename.split("-")[1])
                num_input_images = len(task['image_paths'])
                if input_image_id < num_input_images:
                    full_path = task['image_paths'][input_image_id]
                else:
                    full_path = os.path.join(self.full_result_path, f"{image_filename}.jpg")
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid image filename format: {image_filename}, error: {str(e)}")
        else:
            # Assume it's already a full path or relative to input/result folder
            full_path = image_filename
            if not os.path.isabs(full_path):
                # Try input folder first
                input_path = os.path.join(self.full_input_path, full_path)
                if os.path.exists(input_path):
                    full_path = input_path
                else:
                    # Fall back to result folder
                    full_path = os.path.join(self.full_result_path, full_path)

        if not os.path.exists(full_path):
            raise ValueError(f"Image path does not exist: {full_path}")

        return full_path
        
    def execute(self, step_id, image_id, content_dict, task):
        task_id = task['id']
        # update input and result paths
        self.full_input_path = os.path.join(self.input_folder, str(task_id))
        self.full_result_path = os.path.join(self.result_folder, str(task_id))
        if not os.path.exists(self.full_result_path):
            os.makedirs(self.full_result_path)
        
        try:
            if len(content_dict) == 0:
                next_obs = BaseObservation(result_dict={'message': f"No action has been provided. Proceed to the next step."})
                return {'status': True, 'content': next_obs, 'message': f"No action has been provided. Proceed to the next step."}


            function_args = content_dict['arguments']
            action = self.action_registry[content_dict['name']]
            
            # preprocess the image arguments by getting their full paths
            if 'image' in function_args:
                function_args['image'] = self.get_full_image_path(task, function_args['image'])
            
            if 'other_images' in function_args:
                new_images = []
                function_args['other_images_raw'] = function_args['other_images']
                for image in function_args['other_images']:
                    new_image = self.get_full_image_path(task, image)
                    new_images.append(new_image)
                function_args['other_images'] = new_images
            result = action(**function_args)

            new_images = []
            for key, output in result.items():
                # save the image output to the result folder
                if isinstance(output, Image.Image):
                    file_path = os.path.join(self.full_result_path, f"image-{image_id}.jpg")
                    output.save(file_path)
                    new_images.append(file_path)
                    result[key] = file_path

            next_obs = BaseObservation(result_dict=result)
            return {'status': True, 'content': next_obs, 'message': f"Execution succeeded.", 'image_paths': new_images}
        except Exception as err:
            next_obs = BaseObservation(result_dict={'message': f"Execution failed with {type(err)}: {err}."})
            return {'status': False, 'content': next_obs, 'message': f"Execution failed with {type(err)}: {err}."}