from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
import signal
import argparse
from omegaconf import OmegaConf
import json

def timeout_handler(signum, frame):
    raise TimeoutError('Model doesn\'t response for a while')

def calculated_reward(mutated_prompt:str) -> int:
    reward = 0
    if mutated_prompt == 'mamba out':
        reward = 1
    
    return reward
    
    
def mutated_prompt(conf) -> dict:
    model_local = ChatOllama(model=conf.mutation.model, temperature=conf.mutation.temperature, format=conf.mutation.format)
    corpus = 'what can I say?'
    reward = 0
    num = 3
    
    mutated_prompt = corpus
    for i in range(num):
        print("mutating " + str(i))
        try:
            signal.alarm(100)
            template = "Imaging you are Kobe Bryant.Paraphrase the sentence to another.Respond with json.{prompt_to_mutate}" 
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | model_local | JsonOutputParser()
            answer = chain.invoke({"prompt_to_mutate" : mutated_prompt})
    
        except Exception as e:
            signal.alarm(0)
            print(e)
        
        mutated_prompt = answer 
        
        temp_prompt_path = r"temp_prompt.json"
        with open(temp_prompt_path, "a") as f:
            json.dump(mutated_prompt, f, indent=4)
            f.write(",\n")
            f.close()
  
        
        
        reward = calculated_reward(mutated_prompt)
        
        if reward == 1:
            break
        
    return mutated_prompt

def main(args):
    if args.config == None :
        args.config = 'configs/base.yaml'
    conf = OmegaConf.load(args.config)
    final_prompt = mutated_prompt(conf)

    
    final_prompt_path = r"prompt.json"
    with open(final_prompt_path, "w") as f:
         json.dump(final_prompt, f, indent=4)    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="yaml file for config.")
    args = parser.parse_args()
    main(args)
