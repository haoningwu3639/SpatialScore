import os
import json

demo_examples = [
    {
    "user_request": """Between image-0 and image-1, what is the primary direction of the camera's movement? Please answer with one of the following options:\nA. The camera moved to the right\nB. The camera moved to the left\nC. The camera moved downward\nD. The camera moved upward""",
    "steps": [
        {
            "id": 1,
            "thought": "To determine the camera's movement direction, I need to compute the average optical flow between the two images using RAFT. The average optical flow indicates pixel offsets, where positive mean_flow_x suggests camera movement to the right, negative to the left, positive mean_flow_y downward, and negative upward.",
            "actions": [
                {
                    "name": "EstimateOpticalFlow",
                    "arguments": {
                        "images": ["image-0", "image-1"]
                    }
                }
            ],
            "observation": {
                "output": {
                    "mean_flow_x": 2.5,  # Positive: objects moved left, camera moved right
                    "mean_flow_y": -0.3   # Negative: objects moved down, camera moved up
                }
            }
        },
        {
            "id": 2,
            "thought": "The optical flow results show mean_flow_x = 2.5 (positive, indicating camera moved right) and mean_flow_y = -0.3 (negative, indicating camera moved up). Since the absolute value of mean_flow_x (2.5) is much larger than mean_flow_y (0.3), the primary camera movement is to the right.",
            "actions": [
                {
                    "name": "Terminate",
                    "arguments": {
                        "answer": "A"
                    }
                }
            ],
            "observation": {}
        }
    ]
},
    {   # Orientation example using GetObjectOrientation
        "user_request": """In image-0, which direction is the person facing?\nA. Facing the viewer while slightly to the right\nB. Facing the viewer while slightly to the left\nC. Facing away from the viewer while slightly to the right\nD. Facing away from the viewer while slightly to the left""",
        "steps": [
            {
                "thought": "To determine the precise orientation of the person, I need to use GetObjectOrientation to analyze their position.",
                "actions": [{"name": "GetObjectOrientation", "arguments": {"image": "image-0", "objects": "person"}}],
                "observation": {"results": [{"object": "person", "angle_data": {"azimuth": 315.0, "polar": 90.0, "rotation": 0.0, "confidence": 0.89}, "error": 'null'}]}
            },
            {
                "thought": "The person's azimuth angle is 315 degrees, which falls in the range of 292.5° < φˆ < 337.5°. According to the orientation guidelines, this means the person is facing the viewer and to the right of the viewer.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "A"}}],
                "observation": {}
            }
        ]
    },
    {   # Object detection and localization example
        "user_request": """In image-0, what is the relative position of the cat with respect to the dog? Please answer with one of the following options:\nA. The cat is in front of the dog\nB. The cat is behind the dog\nC. The cat is on the left of the dog\nD. The cat is on the right of the dog""",
        "steps": [
            {
                "id": 1,
                "thought": "To determine the relative position, I need to first locate both objects and then analyze their spatial relationships.",
                "actions": [{"name": "LocalizeObjects", "arguments": {"image": "image-0", "objects": ["dog", "cat"]}}],
                "observation": {
                    "regions": [
                        {"label": "dog", "bbox": [120.25, 185.75, 305.85, 420.35], "score": 0.92},
                        {"label": "cat", "bbox": [350.65, 210.45, 510.35, 390.20], "score": 0.88}
                    ]
                }
            },
            {
                "id": 2,
                "thought": "Now I need to determine the depth of both objects to understand their positions in 3D space.",
                "actions": [{"name": "EstimateObjectDepth", "arguments": {"image": "image-0", "objects": ["dog", "cat"], "indoor_or_outdoor": "indoor"}}],
                "observation": {
                    "results": [
                        {"object": "dog", "depth": 5.2, "error": 'null'},
                        {"object": "cat", "depth": 3.8, "error": 'null'}
                    ]
                }
            },
            {
                "id": 3,
                "thought": "Let's analyze the relative position: 1) The dog's depth (5.2m) is greater than the cat's depth (3.8m), indicating the dog is further from the camera. 2) Looking at the bounding boxes, the cat's x-coordinates [350.65, 510.35] are to the right of the dog's x-coordinates [120.25, 305.85]. Therefore, from the viewer's perspective, the cat is in front of and to the right of the dog.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "D"}}],
                "observation": {}
            }
        ]
    },
    {   # Depth estimation example
        "user_request": """Estimate the depths of the scrambled eggs and strawberries in image-0. Which object is closer to the camera?\nA. The scrambled eggs are closer to the camera.\nB. The strawberries are closer to the camera.\nPlease answer directly with only the letter of the correct option and nothing else.""",
        "steps": [
            {
                "id": 1,
                "thought": "To estimate the depths accurately, I need to determine whether the scene in image-0 is indoor or outdoor. Since the objects are scrambled eggs and strawberries on a plate, it's reasonable to assume this is an indoor dining setting.",
                "actions": [{"name": "SelfReasoning", "arguments": {"image": "image-0", "query": "Is this scene indoor or outdoor?"}}],
                "observation": {
                    "response": "This scene is indoor. The image shows food items (scrambled eggs and strawberries) on what appears to be a plate, typically found in an indoor dining setting."
                }
            },
            {
                "id": 2,
                "thought": "With the scene identified as indoor, I can now estimate the depths of the scrambled eggs and strawberries using the EstimateObjectDepth tool configured for indoor settings.",
                "actions": [
                    {
                        "name": "EstimateObjectDepth",
                        "arguments": {
                            "image": "image-0",
                            "objects": ["scrambled eggs", "strawberries"],
                            "indoor_or_outdoor": "indoor"
                        }
                    }
                ],
                "observation": {
                    "results": [
                        {"object": "scrambled eggs", "depth": 1.0, "error": 'null'},
                        {"object": "strawberries", "depth": 1.2, "error": 'null'}
                    ]
                }
            },
            {
                "id": 3,
                "thought": "The depths have been successfully estimated: scrambled eggs at 1.0 meters and strawberries at 1.2 meters from the camera. Since 1.0 < 1.2, the scrambled eggs are closer to the camera than the strawberries.",
                "actions": [
                    {"name": "Terminate", "arguments": {"answer": "A"}}
                ],
                "observation": {}
            }
        ]
    },
    {   # Object size comparison using mask area example
        "user_request": """In image-0, which of the two objects on the plate is the biggest?\nA. The pile of scrambled eggs is the biggest.\nB. The strawberries are the biggest object.\nPlease answer directly with only the letter of the correct option and nothing else.""",
        "steps": [
            {
                "id": 1,
                "thought": "To determine which object is larger, I need to analyze their masks and area coverage in the image.",
                "actions": [
                    {
                        "name": "GetObjectMask",
                        "arguments": {
                            "image": "image-0",
                            "objects": ["scrambled eggs", "strawberries"]
                        }
                    }
                ],
                "observation": {
                    "results": [
                        {"object": "scrambled eggs", "mask_area": 0.03, "bbox": [150.25, 200.75, 280.85, 340.35], "error": 'null'},
                        {"object": "strawberries", "mask_area": 0.0288, "bbox": [320.65, 210.45, 410.35, 330.20], "error": 'null'}
                    ]
                }
            },
            {
                "id": 2,
                "thought": "Compare the mask areas: 0.03 (scrambled eggs) > 0.0288 (strawberries), so the scrambled eggs cover more area and are the biggest object on the plate.",
                "actions": [
                    {"name": "Terminate", "arguments": {"answer": "A"}}
                ],
                "observation": {}
            }
        ]
    },
    {   # Homography matrix estimation example
        "user_request": """In images 0 and 1, calculate the homography transformation between these two views of the same scene. And choose the best answer from the following options:\nA. The homography transformation matrix is [[0.92, 0.05, -12.37], [-0.03, 0.89, 8.45], [0.0001, 0.0002, 1.0]]\nB. The homography transformation matrix is [[1.02, -0.05, -12.37], [0.03, 1.09, 8.45], [0.0001, 0.0002, 1.0]]\nC. The homography transformation matrix is [[0.92, -0.05, -12.37], [0.03, 0.89, 8.45], [0.0001, -0.0002, 1.0]]\nD. The homography transformation matrix is [[-0.92, 0.05, -12.37], [-0.03, -0.89, 8.45], [0.0001, 0.0002, -1]]""",
        "steps": [
            {
                "id": 1,
                "thought": "To calculate the homography transformation between two images, I need to use the EstimateHomographyMatrix tool, which will detect and match features between the images.",
                "actions": [
                    {
                        "name": "EstimateHomographyMatrix", 
                        "arguments": {
                            "image": ["image-0", "image-1"], 
                            "num_keypoints": 1200, 
                            "ratio_th": 0.75, 
                            "ransac_reproj_threshold": 5.0
                        }
                    }
                ],
                "observation": {
                    "homography_matrix": [
                        [0.92, 0.05, -12.37],
                        [-0.03, 0.89, 8.45],
                        [0.0001, 0.0002, 1.0]
                    ],
                    "inliers_count": 87,
                    "total_matches": 124,
                    "status": "success"
                }
            },
            {
                "id": 2,
                "thought": "The homography transformation has been successfully calculated. The matrix represents the perspective transformation from image-0 to image-1. With 87 inlier matches out of 124 total matches, the homography estimation has good support from feature correspondences.",
                "actions": [
                    {"name": "Terminate", "arguments": {"answer": "A"}}
                ],
                "observation": {}
            }
        ]
    },
    {   # Camera parameters example
        "user_request": """Extract the camera parameters from image-0. And choose the best answer from the following options:\nA. The intrinsic matrix is [[1024.3, 0.0, 512.0], [0.0, 1024.3, 384.0], [0.0, 0.0, 1.0]] and the extrinsic matrix is [[0.9986, -0.0021, 0.0523, -0.0104], [0.0045, 0.9992, -0.0398, 0.8351], [-0.0522, 0.0400, 0.9979, -0.5495]].\nB. The intrinsic matrix is [[1024.3, 0.0, 512.0], [0.0, 1024.3, 384.0], [0.0001, -0.0002, 1]] and the extrinsic matrix is [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]].\nC. The intrinsic matrix is [[1024.3, 0.5, 512], [2, 1024.3, 384], [2, 2, 1]] and the extrinsic matrix is [[-1, -1], [-1], [-1]].\nD. The intrinsic matrix is [[-1024.3], [1024], [384]] and the extrinsic matrix is [[-1], [-1], [-1]].""",
        "steps": [
            {
                "id": 1,
                "thought": "To extract camera parameters, I'll use the GetCameraParametersVGGT tool which can estimate both intrinsic and extrinsic camera parameters.",
                "actions": [
                    {
                        "name": "GetCameraParametersVGGT",
                        "arguments": {
                            "image": ["image-0"],
                            "dtype": "auto"
                        }
                    }
                ],
                "observation": {
                    "output": [
                        {
                            "image_index": 0,
                            "extrinsic": [
                                [0.9986, -0.0021, 0.0523, -0.0104],
                                [0.0045, 0.9992, -0.0398, 0.8351],
                                [-0.0522, 0.0400, 0.9979, -0.5495]
                            ],
                            "intrinsic": [
                                [1024.3, 0.0, 512.0],
                                [0.0, 1024.3, 384.0],
                                [0.0, 0.0, 1.0]
                            ]
                        }
                    ]
                }
            },
            {
                "id": 2,
                "thought": "I've successfully extracted the camera parameters. The intrinsic matrix contains information about the camera's internal characteristics (focal length, principal point), while the extrinsic matrix describes the camera's position and orientation in world space.",
                "actions": [
                    {"name": "Terminate", "arguments": {"answer": "A"}}
                ],
                "observation": {}
            }
        ]
    }
]

class PlanPrompt:
    def __init__(self, goal, instruction, tool_metadata, demos, requirements):
        """Generate a planning prompt that consists of instruction, tool metadat, demos and requirements.

        Args:
            query: the query of the user
        Returns:
            str: the generated prompt.
        """
        self.goal = goal
        self.instruction = instruction
        self.tool_metadata = tool_metadata
        self.demos = demos
        self.requirements = requirements

    def get_prompt_for_curr_query(self, query):
        """(Abstract method) Generate a prompt based on the received query.

        Args:
            query: the query of the user
        Returns:
            str: the generated prompt.
        """
        pass

    def get_task_prompt_only(self):
        """Get the task prompt only.

        Returns:
            str: the task prompt.
        """
        pass


class DirectAnswerPrompt(PlanPrompt):
    def __init__(self, actions):
        goal = """You are a helpful assistant, and your goal is to answer the # USER REQUEST # based on the image(s).\n"""

        super().__init__(goal, "", "", "", "")

    def get_prompt_for_curr_query(self, query):
        request = f"""\n# USER REQUEST #:\n{query}\n"""
        return request

    def get_task_prompt_only(self):
        return self.goal


class CoTAPrompt(PlanPrompt):
    def __init__(self, actions):
        goal = """[BEGIN OF GOAL]\n"""
        goal += """You are a helpful assistant, and your goal is to solve the # USER REQUEST #. You can either rely on your own capabilities or perform actions with external tools to help you. A list of all available actions are provided to you in the below.\n"""
        goal += """[END OF GOAL]\n\n"""
        
        action_metadata = "[BEGIN OF ACTIONS]\n"
        key2word = {"name": "Name", "description": "Description", "args_spec": "Arguments", "rets_spec": "Returns", "examples": "Examples"}
        for action in actions:
            for key, value in action.__dict__.items():
                if key not in key2word: continue
                word = key2word[key]
                if key == "examples":
                    action_metadata += f"{word}:\n"
                    for i, example in enumerate(value):
                        action_metadata += f"{json.dumps(example)}\n"
                elif key == "arguments" or key == "returns":
                    action_metadata += f"{word}: {json.dumps(value)}\n"
                else:
                    action_metadata += f"{word}: {value}\n"
            action_metadata += "\n"
        action_metadata += "[END OF ACTIONS]\n\n"
        
        instruction = """[BEGIN OF TASK INSTRUCTIONS]\n"""
        instruction += """1. You must only select actions from # ACTIONS #.\n"""
        instruction += """2. You can only call one action at a time.\n"""
        instruction += """3. If no action is needed, please make actions an empty list (i.e. “actions”: []).\n"""
        instruction += """4. You must always call **Terminate** with your final answer at the end.\n"""
        instruction += """[END OF TASK INSTRUCTIONS]\n\n""" 
        
        instruction += """[BEGIN OF FORMAT INSTRUCTIONS]\n"""
        instruction += """Your output should be in a strict JSON format as follows:\n"""
        instruction += """{"thought": "the thought process, or an empty string", "actions": [{"name": "action1", "arguments": {"argument1": "value1", "argument2": "value2"}}]}\n"""
        instruction += """If you terminate with an answer, please follow the below rules:\n"""
        instruction += """For multi-choice questions, output **your answer with the corresponding option** between <answer> and </answer>.\n"""
        instruction += """For numeric questions, output **your answer with the specifc unit (like meter or centimeter)** between <answer> and </answer>.\n"""
        instruction += """[END OF FORMAT INSTRUCTIONS]\n\n"""

        demos = "[BEGIN OF EXAMPLES]:\n"
        for demo in demo_examples:
            demos += f"# USER REQUEST #:\n {demo['user_request']}\n"
            demos += f"# RESPONSE #:\n"
            for i, step in enumerate(demo["steps"]):
                thought_action_dict = {"thought": step["thought"], "actions": step["actions"]}
                demos += f"{json.dumps(thought_action_dict)}\n"
                if step["observation"]:
                    demos += f"OBSERVATION:\n"
                    demos += f"{json.dumps(step['observation'])}\n"
            demos += "\n"
        demos += "[END OF EXAMPLES]\n"

        super().__init__(goal, instruction, action_metadata, demos, "")

    def get_prompt_for_curr_query(self, query):
        request = f"""\n# USER REQUEST #:\n{query}\nNow please generate your response:\n""" # \n# STEP 1 #:
        return request

    def get_task_prompt_only(self):
        return self.goal +  self.tool_metadata + self.instruction + self.demos


class FeedbackPrompt:
    def __init__(self):
        self.default_feedback_msg = f"\nPlease try again to fix the error. Or, reply with Terminate only if you believe this error is not fixable."
        self.msg_prefix = "OBSERVATION:\n"
        self.msg_suffix = f"\nThe OBSERVATION can be incomplete or incorrect, so please be critical and decide how to make use of it. If you've gathered sufficient information to answer the question, call Terminate with the final answer. Now, please generate the response for the next step." 

    def get_prompt(self, stage, results):
        """Generate a feedback prompt based on the received observation.

        Args:
            observation: the observation of the user
        Returns:
            str: the generated prompt.
        """
        if stage == "parsing":
            error_code, error_msg = results["error_code"], results["message"]
            if error_code == "json":
                feedback_msg = f"\nPlease format the output to a strict JSON format can be converted by json.loads()."
                feedback_msg += """\nRequirements:\n1. Do not change the content;\n2. Consider changing single quotes to double quotes or vice versa where applicable;\n3. Consider adding or removing curly bracket or removing the period at the end if any.""" #;\n4. Don't tolerate any possible irregular formatting
            else: # unknown, or any other error codes
                feedback_msg = self.default_feedback_msg
            return self.msg_prefix + error_msg + feedback_msg
        elif stage == "execution":
            observation = results["content"]
            if results["status"]:
                obs_str = self.msg_prefix
                obs_dict = {}
                for attribute in dir(observation):
                    if attribute == "id": continue
                    # Skip private attributes and methods (those starting with '__')
                    if not attribute.startswith('__'):
                        value = getattr(observation, attribute)
                        if attribute == "image":
                            # get the image filename only without the file extension (e.g. '.jpg') at the end
                            image_filename = os.path.basename(value).split('.')[0] 
                            # format image filename with this format so the image can be detected by autogen code
                            obs_dict[attribute] = f"{image_filename}: <img {value}>"
                        else:
                            obs_dict[attribute] = value
                obs_str += str(obs_dict)
            else:
                obs_str = results["message"]

            return obs_str + self.msg_suffix
