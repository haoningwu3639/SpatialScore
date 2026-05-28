import os
import json

demo_examples = [
        {
            "user_request": """Between image-0 and image-1, what is the primary direction of the camera's movement? Please answer with one of the following options:\nA. The camera moved to the right\nB. The camera moved to the left\nC. The camera moved downward\nD. The camera moved upward""",
            "steps": [
                {
                    "id": 1,
                    "thought": "To determine the camera's movement direction, I need to compute the average optical flow between the two images using RAFT. The average optical flow indicates pixel offsets, where positive mean_flow_x suggests camera movement to the right, negative to the left, positive mean_flow_y downward, and negative upward.",
                    "actions": [{"name": "EstimateOpticalFlow", "arguments": {"images": ["image-0", "image-1"]}}],
                    "observation": {"output": {"mean_flow_x": 2.5, "mean_flow_y": -0.3}} # Positive: objects moved left, camera moved right; Negative: objects moved down, camera moved up
                },
                {
                    "id": 2,
                    "thought": "The optical flow results show mean_flow_x = 2.5 (positive, indicating camera moved right) and mean_flow_y = -0.3 (negative, indicating camera moved up). Since the absolute value of mean_flow_x (2.5) is much larger than mean_flow_y (0.3), the primary camera movement is to the right.",
                    "actions": [{"name": "Terminate", "arguments": {"answer": "(A)"}}],
                    "observation": {}
                }
            ]
        },
        # Orientation
        {   
            "user_request": """In image-0, which direction is the person facing?\nA. Facing the viewer while slightly to the right\nB. Facing the viewer while slightly to the left\nC. Facing away from the viewer while slightly to the right\nD. Facing away from the viewer while slightly to the left""",
            "steps": [
                {
                    "thought": "To determine the precise orientation of the person, I need to use GetObjectOrientation to analyze their position.",
                    "actions": [{"name": "GetObjectOrientation", "arguments": {"image": "image-0", "objects": "person"}}],
                    "observation": {"results": [{"object": "person", "angle_data": {"azimuth": 315.0, "polar": 90.0, "rotation": 0.0, "confidence": 0.89}, "error": 'null'}]}
                },
                {
                    "thought": "The person's azimuth angle is 315 degrees, which falls in the range of 292.5° < φˆ < 337.5°. According to the orientation guidelines, this means the person is facing the viewer and to the right of the viewer.",
                    "actions": [{"name": "Terminate", "arguments": {"answer": "(A)"}}],
                    "observation": {}
                }
            ]
        },
        {
        "user_request": """In image-0, what is the distance between the mug and the laptop?\nA. 15 centimeters\nB. 29 centimeters\nC. 45 centimeters\nD. 60 centimeters""",
        "steps": [
            {
                "thought": "First, I need to determine whether the scene in image-0 is indoors or outdoors. Observing the image, I see a desk, a mug, and a laptop in an office setting, indicating an indoor scene. Since it is indoors, according to the instruction, I do not need to call any tools and can directly estimate the distance.",
                "actions": [],
                "observation": {}
            },
            {
                "thought": "Based on the visual cues in the image, the distance between the mug and the laptop is approximately 29 centimeters.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "(B)"}}],
                "observation": {}
            }
        ]
    },
    {
        "user_request": """In image-0, which object on the desk is larger in size?\nA. The mug\nB. It can not be decided given the image only\nC. The laptop\nD. They seem to be of almost the same size""",
        "steps": [
            {
                "thought": "First, I need to determine whether the scene in image-0 is indoors or outdoors. Observing the image, I see a desk with a mug and a laptop in an office setting, indicating an indoor scene(For object size questions (including length, width, height, tall, short, slim, or heavy), do NOT call any tools. You should answer directly based on the visual information from the image.). Since it is indoors, according to the instruction, I do not need to call any tools and can directly estimate the relative size of the objects.",
                "actions": [],
                "observation": {}
            },
            {
                "thought": "Based on the visual cues in the image, the laptop is clearly larger in size compared to the mug.",
                "actions": [{"name": "Terminate", "arguments": {"answer": "(C)"}}],
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
        goal += """You are a helpful assistant, and your goal is to solve the # USER REQUEST #. You can either rely on your own capabilities or perform actions with external tools to help you. A list of all available actions is provided to you below.\n"""
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
        instruction += """5. Please note that the priority of the SelfThinking tool is relatively low. Please give priority to using other tools, and only consider using this tool if the problem cannot be solved otherwise.\n"""
        instruction += """[END OF TASK INSTRUCTIONS]\n\n""" 

        instruction = """[BEGIN OF TOOL USAGE INSTRUCTIONS]\n"""
        instruction += """1. **Construct the correct image path** for the tool to use, ensuring the path can be accessed and read properly.\n"""
        instruction += """2. For object distance and object size(Length, width, height,tall, short, slim, or heavy) problems, first observe the image. 
If the scene is outdoors, **FIRST** use `LocalizeObjects` to obtain the 2D bounding boxes, then determine the pair of points (one from each object) that are closest to each other, and use these points as the 'point' inputs for `Get3DDistance` to get the distance between the two objects. 
Do **NOT** simply use the center points of the boxes as the closest points between two objects.\nDo **NOT** simply use the center points of the boxes as the closest points between two objects.\nDo **NOT** simply use the center points of the boxes as the closest points between two objects.\n"""
        instruction += """3. For counting-related problems, **USE** `CountObjects`; the number of returned points equals the number of objects.\n"""
        instruction += """4. For camera-related problems, you may need to **USE** `GetCameraParametersVGGT` to obtain the camera parameters.\n"""
        instruction += """\n"""
        instruction += """==================== ⚠️ CRITICAL WARNING ⚠️ ====================\n"""
        instruction += """**DO NOT** invent or mention any tool that is **NOT explicitly defined** in #ACTIONS#.\n"""
        instruction += """**DO NOT** fabricate tool usage results if you have NOT actually called the tool.\n"""
        instruction += """You MUST only describe tool results that are actually obtained during execution.\n"""
        instruction += """Violation of this rule is considered a **SERIOUS ERROR**.\n"""
        instruction += """===============================================================\n"""
        instruction += """==================== ⚠️ RELIABILITY WARNING ⚠️ =================\n"""
        instruction += """If a tool result contains **ambiguous references** — for example, `LocalizeObjects` returns multiple bounding boxes for the same object — **the result is unreliable**.\n"""
        instruction += """In such cases, you SHOULD rely on **reasoning** instead of depending on the tool output.\n"""
        instruction += """Treat this as a high-risk situation and avoid making decisions solely based on such tool results.\n"""
        instruction += """===============================================================\n"""
        instruction += """==================== ⚠️ TOOL CHAIN LENGTH WARNING ⚠️ ============\n"""
        instruction += """If the tool invocation chain becomes **too long**, you MUST **STOP** calling further tools to avoid reaching the maximum number of allowed calls.\n"""
        instruction += """In such cases, immediately switch to using **SelfThinking** to answer, **INCLUDING all input images** required for reasoning.\n"""
        instruction += """Failure to follow this rule may result in task termination without producing a valid answer.\n"""
        instruction += """===============================================================\n"""
        instruction += """[END OF TOOL USAGE INSTRUCTIONS]\n\n"""


        instruction += """[BEGIN OF FORMAT INSTRUCTIONS]\n"""
        instruction += """Your output should be in a strict JSON format as follows:\n"""
        instruction += """{"thought": "the thought process, or an empty string", "actions": [{"name": "action1", "arguments": {"argument1": "value1", "argument2": "value2"}}]}\n"""
        # instruction += """If you terminate with an answer, please follow the below rules:\n"""
        # instruction += """For multi-choice questions, output **your answer with the corresponding option** between <answer> and </answer>.\n"""
        # instruction += """For numeric questions, output **your answer with the specifc unit (like meter or centimeter)** between <answer> and </answer>.\n"""
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
        self.default_feedback_msg = f"\nPlease try again to fix the error. Or, reply with **Terminate** only if you believe this error is not fixable."
        self.msg_prefix = "OBSERVATION:\n"
        self.msg_suffix = f"\nThe OBSERVATION can be incomplete or incorrect, so please be critical and decide how to make use of it. If you've gathered sufficient information to answer the question, call **Terminate** with the final answer. Now, please generate the response for the next step." 

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
