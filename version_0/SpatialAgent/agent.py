from typing import Dict, Optional, Union
from utils.prompt import CoTAPrompt, DirectAnswerPrompt
from autogen.agentchat import Agent, UserProxyAgent

class UserAgent(UserProxyAgent):
    def __init__(self, name, prompt_generator, feedback_generator, parser, executor, **config):
        super().__init__(name, **config)
        self.prompt_generator = prompt_generator
        self.feedback_generator = feedback_generator
        self.parser = parser
        self.executor = executor
        self.step_id = 0
        self.current_image_id = 0
        self.called_tools = []
        self.new_image_paths = []

    def sender_hits_max_reply(self, sender: Agent):
        return self._consecutive_auto_reply_counter[sender.name] >= self._max_consecutive_auto_reply

    def generate_init_message(self, query):
        content = self.prompt_generator.get_prompt_for_curr_query(query)
        return content

    def initiate_chat(self, assistant, message, task, image_detail='auto'):
        self.step_id = 0
        self.task = task
        self.current_image_id = len([task['image_paths']]) # current image id is the number of input images        
        self.called_tools = []
        self.new_image_paths = []
        self.final_answer = None
        
        initial_message = self.generate_init_message(message)
        initial_message = {"content": initial_message, "role": "user", "detail": image_detail}

        assistant.receive(initial_message, self, request_reply=True)

    def receive(self, message: Union[Dict, str], sender: Agent, request_reply=False, silent: Optional[bool] = False):
        """ Receive a message from the sender agent. """
        print("COUNTER:", self._consecutive_auto_reply_counter[sender.name])
        self._process_received_message(message, sender, silent)
        
        parsed_results = self.parser.parse(message)
        parsed_content = parsed_results['content']
        parsed_status = parsed_results['status']
        
        # Terminate planning if parsing fails and termination check returns true
        # Otherwise proceed to parsing feedback or verification/execution
        if not parsed_status and self._is_termination_msg(message):
            return

        if parsed_status:
            if isinstance(self.parser.prompt_generator, DirectAnswerPrompt):
                self.final_answer = parsed_content
                self._consecutive_auto_reply_counter[sender.name] = 0
                return

            elif isinstance(self.parser.prompt_generator, CoTAPrompt):
                if len(parsed_content) > 0: # actions are called
                    action_name = parsed_content['name'] 
                    if action_name == "Terminate":
                        if "answer" in parsed_content['arguments']:
                            self.final_answer = parsed_content['arguments']['answer']
                    self.called_tools += [parsed_content]
                
                print("Called tools:", self.step_id, self.current_image_id, parsed_content, self.task)
                
                executed_results = self.executor.execute(self.step_id, self.current_image_id, parsed_content, self.task)
                executed_status = executed_results['status']

                if executed_status and 'image_paths' in executed_results:
                    self.new_image_paths += executed_results['image_paths']

                if self.sender_hits_max_reply(sender) or self._is_termination_msg(message):
                    # reset the consecutive_auto_reply_counter
                    self._consecutive_auto_reply_counter[sender.name] = 0
                    return
                
                # send the message from running the execution module
                self._consecutive_auto_reply_counter[sender.name] += 1
                
                feedback_msg = self.feedback_generator.get_prompt("execution", executed_results)
                if executed_status and getattr(executed_results['content'], 'image', None):
                    self.current_image_id += 1
                self.step_id += 1
                self.send(feedback_msg, sender, request_reply=True)  
        else:
            if self.sender_hits_max_reply(sender) or self._is_termination_msg(message):
                self._consecutive_auto_reply_counter[sender.name] = 0
                return
            self._consecutive_auto_reply_counter[sender.name] += 1
            feedback_msg = self.feedback_generator.get_prompt("parsing", parsed_results)
            self.step_id += 1
            self.send(feedback_msg, sender, request_reply=True)
