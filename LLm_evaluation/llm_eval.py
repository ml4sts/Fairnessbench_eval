import openai
from config import API_KEY  # Import the API key from the config file

openai.api_key = API_KEY

 
def read_code(file_path: str):
        with open(file_path, 'r') as file:
             code = file.read()
        return code

def llm_evaluation(code: str, prompt_template: str):
        prompt = prompt_template.replace("[Insert Code Here]", code)
        print(prompt)
        response = openai.chat.completions.create(
            model="gpt-4.1",  
            messages=[{
               "role": "system", 
               "content": "You are an assistant that evaluates code based on fairness rubric."
            }, {
               "role": "user", 
               "content": prompt
            }],
            max_tokens=500,  
            temperature=0.7,  
        )
    
        return response.choices[0].message.content

file_path = 'train_updated.py'  # Path to the file containing the code

    # Read the code from the file
code = read_code(file_path)

    # Read the prompts from a file (for evaluation)
prompt_file = 'prompt.txt'  # Path to the file containing multiple prompts

    # Function to read the prompts from a file
def read_prompts(file_path: str):
        with open(file_path, 'r') as file:
            prompts = file.read().split('---')  # Split by the delimiter (---)
        return prompts

    # Read the prompts
prompts = read_prompts(prompt_file)

    # Loop through all the prompts and evaluate the code with each prompt
for fairness_prompt in prompts:
        result = llm_evaluation(code, fairness_prompt)
        print(result)
