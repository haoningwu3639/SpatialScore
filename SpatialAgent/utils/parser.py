import re
import json
from utils.prompt import CoTAPrompt, DirectAnswerPrompt
    
def extract_outermost_bracket(text):
    # This pattern matches the outermost curly brackets and captures the content inside
    pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}'
    match = re.search(pattern, text)
    if match:
        # Remove the outermost brackets and return the content inside
        return match.group(0)
    else:
        return text
            
class Parser:
    def __init__(self, prompt_generator):
        self.prompt_generator = prompt_generator
    
    def parse(self, response):
        if isinstance(self.prompt_generator, CoTAPrompt):
            return self.parse_cota(response)
        elif isinstance(self.prompt_generator, DirectAnswerPrompt):
            return self.parse_direct_answer(response)
        else:
            raise ValueError(f"Unsupported prompt format: {self.prompt_generator}.")
        
    def parse_direct_answer(self, response):
        if isinstance(response, dict) and 'content' in response:
            content = response['content']
        else:
            content = response
        return {'status': True, 'content': content, 'message': 'Parsing succeeded.', 'error_code': ''}
    
    def parse_cota(self, response):
        if isinstance(response, dict) and 'content' in response:
            content = response['content']
        else:
            content = response
        try:
            # pattern = r'#\s*STEP\s*\d?\s*#\:?' # to match one of these: `# STEP 1 #:`, `# STEP #:`, `#STEP 1#:`, `#STEP#:`, or `#STEP#`
            # match = re.search(pattern, content)
            # if match:
            #     end_index = match.end()
            #     content = content[end_index:]
            if content.find('```json') != -1:
                start_pos = content.find('```json')
                content = content[start_pos+len('```json'):-3]
            content = extract_outermost_bracket(content)
            content = json.loads(content)
            if len(content['actions']) >= 1:
                action = content['actions'][0]
                assert 'name' in action and 'arguments' in action, "missing 'name' or 'arguments' in the parsed action."
            else:
                action = {}
            return {'status': True, 'content': action, 'message': 'Parsing succeeded.', 'error_code': ''}
        except json.JSONDecodeError as err:
            return {'status': False, 'content': content, 'message': f"{type(err)}: {err}.", 'error_code': 'json'} 
        except Exception as err:
            return {'status': False, 'content': content, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}


def main():
    prompt_generator = CoTAPrompt(actions=[])
    parser = Parser(prompt_generator=prompt_generator)
    response = {'content': """                                                                                                                                                                        
    ```json                                                                                                                                                                          
    {                                                                                                                                                                                
        "thought": "First I need to extract the text from the image to analyze and solve the second equation for d.",                                                                  
        "actions": [                                                                                                                                                                   
                {                                                                                                                                                                            
                    "name": "OCR",                                                                                                                                                             
                    "arguments": {"image": "/export/agentstudio-family/zixian/mm-vet/images/v1_1.png"}                                                                                                                                                                          
            }                                                                                                                                                                            
        ]                                                                                                                                                                              
    }                                                                                       
    ```"""}
    result = parser.parse(response)
    print(result)


if __name__ == "__main__":
    main()