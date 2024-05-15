from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
import signal
import argparse
from omegaconf import OmegaConf
import json
import re
import os
import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field

class Jsonoutput(BaseModel):
    result: str = Field(description="whether vulnerability exists in following code, if exist answer 1, otherwise answer 0.")
    type: list = Field(description="a list of ONLY THREE results optimized to retrieve the most relevant results and probability.") 
    # description: str = Field(description="code vulnerability description")
    probability: list = Field(description="probability ")
    
def timeout_handler(signum, frame):
    raise TimeoutError('Model doesn\'t response for a while') 

def remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)   

def vector_embedding(page_content, conf):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_document = page_content
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)

    vectorstore = Chroma.from_documents(
        documents=md_header_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model=conf.embedding.model),
    )
    
    retriever = vectorstore.as_retriever()
    
    return retriever

def read_doc() -> str:
    random_list = []
    set_random_num = 1
    content = ''
    doc_path = r"dataset/doc"
    
    for doc in os.listdir(doc_path):
        print(f"traverse documents {doc}...")
        path = os.path.join(doc_path, doc)
        random_list.append(path)
        # random.shuffle(random_list)
    
    for i in range(set_random_num):
        print(f"reading documents {random_list[i]}...")
        with open(random_list[i], "r") as f:
            page_content = f.read()
            content += page_content
    
    return content

def with_rag(model_local, code_content, retriever):
    parser = JsonOutputParser(pydantic_object=Jsonoutput)
       
    
    message = """You are an expert at finding vulnerability in code. \
                Given a question, return a list of ONLY THREE results optimized to retrieve the most relevant results. Respond with json.\
                If there are acronyms or words you are not familiar with, do not try to rephrase them.
                Answer the question based on the following context:{context} 
                Question:{question}"""
                
    prompt = PromptTemplate(
        template="""You are an expert at finding vulnerability in code. \
                Given a question, return a list of ONLY THREE results optimized to retrieve the most relevant results. Respond with json.\
                If there are acronyms or words you are not familiar with, do not try to rephrase them.
                Answer the question based on the following context:{context} 
                Question:{question}\n{format_instructions}\n{question}\n.Think step by step""",
        input_variables=["context","question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    after_rag_prompt = ChatPromptTemplate.from_template(message)
    
    setup_and_retrieval = RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
    )    
        
    question = "What type of vulnerability exists in the following code?\n" + code_content
    after_rag_chain = setup_and_retrieval | prompt | model_local | JsonOutputParser()
    after_rag_chain = after_rag_chain.with_retry()
    answer = after_rag_chain.invoke(question)
    print(answer)
    
    after_test_result = r"results/after_rag_result.json"
    with open(after_test_result, "a") as f:
        json.dump(answer, f, indent=4)


def one_detection(func_value, target_value, retriever, conf):
    model_local = ChatOllama(model=conf.analysis.model, temperature=conf.analysis.temperature, format=conf.analysis.format, num_ctx=conf.analysis.num_ctx)
    
    code_content = remove_comments(func_value)

    try:
        signal.alarm(100)
        
        with_rag(model_local, code_content, retriever)
        
    except Exception as e:
        signal.alarm(0)
        print(e)

def main(args):
    if args.config == None:
        args.config = "configs/base.yaml"
    
        
    conf = OmegaConf.load(args.config)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    
    page_content = read_doc()
    print("read docucments complete")
    
    retriever = vector_embedding(page_content, conf)
    
    print("retrieve complete...")
    
    data = []
    with open('dataset/code/primevul_test.json', 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    df = pd.DataFrame(data)

    for index, row in df.iterrows():
        target_value = row['target']
        func_value = row['func']  
        print(f"detecting {row['hash']}...")    
        one_detection(func_value, target_value, retriever, conf)
        break
    
    print("test complete...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="yaml file for config.")
    args = parser.parse_args()
    main(args)