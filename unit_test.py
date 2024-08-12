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
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader

class Jsonoutput(BaseModel):
    Vulnerability_Detection: str = Field(description="If this code snippet has vulnerabilities, output Yes; otherwise, output No.")
    Vulnerability_Assessment: list = Field(description="a list of ONLY THREE results optimized to retrieve the most relevant results of the code snippet, and give more star numbers if the answer is more probable.") 
    Vulnerability_Location: str = Field(description="Provide a vulnerability location result for the vulnerable code snippet.") 

class Argueoutput(BaseModel):
    Vulnerability_Detection: str = Field(description="If this code snippet has vulnerabilities, output Yes; otherwise, output No.")
    # select_answer: str = Field(description="select the most probable answer from the Vulnerability_Assessment")
    select_answer: str = Field(description="whether a specific type of vulnerability exists in the code and summarizations")
    
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
    pdf_flag = 0
    if pdf_flag == 1:
        splits = page_content
    
    else:
        headers_to_split_on = [
            ("#", "Header 1")
        ]
        
        markdown_document = page_content
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits = markdown_splitter.split_text(markdown_document)

    vectorstore = Chroma.from_documents(
        documents=splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(base_url=conf.embedding.base_url, model=conf.embedding.model),
    )
    
    retriever = vectorstore.as_retriever()
    
    return retriever

def read_pdf() -> str:
    loader = PyPDFLoader("dataset/doc/cwe_latest.pdf")
    pages = loader.load_and_split()
    
    return pages
    

def read_doc() -> str:
    
    content = ''
    doc_path = r"dataset/doc/cwe_latest.md"
    
    print(f"reading documents ...")
    with open(doc_path, "r") as f:
        page_content = f.read()
        content += page_content
    
    return content


def with_rag(model_local, code_content, retriever):
    parser = JsonOutputParser(pydantic_object=Jsonoutput)
                
    prompt = PromptTemplate(
        template="""You are an expert at finding vulnerability in code. \
                Given a question, return a list of ONLY THREE results optimized to retrieve the most relevant results. Respond with json.\
                If there are acronyms or words you are not familiar with, do not try to rephrase them.
                Answer the question based on the following context:{context} 
                Question:{question}\n{format_instructions}\n{question}\n.Think step by step""",
        input_variables=["context","question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    setup_and_retrieval = RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
    )    
        
    question = "What type of vulnerability exists in the following code?\n" + code_content + "Generate more star if you think the answer is more probable."
    after_rag_chain = setup_and_retrieval | prompt | model_local | JsonOutputParser()
    after_rag_chain = after_rag_chain.with_retry()
    answer = after_rag_chain.invoke(question)
    
    print(answer)
    vulnerability_list = dict.get(answer, "Vulnerablitity_Assessment")
    first_result = dict.get(answer, "Vulnerability_Detection")
    #no dict slice, or raise error.
    
    message = """1) If one reason describes code that does not exist in the provided input, it is not valid.
                 2) If one reason is not related to the code, the reason is not valid.
                 3) If this reason violates the facts, the reason is unreasonable.
                 4) If one reason is not related to the decision, the reason is not valid.
                 5) If one reason assume any information that is not provided, the reason is not valid.
                 6) If the code is safe and one reason supports the decision, please check if the code has other potential vulnerabilities. If the code has other potential vulnerabilities, the reason is not valid.
                 7) The selected reason should be the most relevant to the decision.
                 8) The selected reason must be the most reasonable and accurate one.
                 9) The selected reason must be factual, logical and convincing.
                 10) Do not make any assumption out of the given code"""
    
    parser = JsonOutputParser(pydantic_object=Argueoutput)
    
    query = "You are a code auditing expert. I will give you a code and some summarizations from other experts. Your task is to answer whether a specific type of vulnerability exists in the code and summarizations following the specific instrutions below.Here is the code:\n"
    query += "{format_instructions}\n"
    query += """{code}\n"""
    query += "Summarizations:\n"
    query += "{answer}"
    query += "Instructions:\n"
    query += "{message}\n"
    
    
    prompt = PromptTemplate(
        template = query,
        input_variables=["code", "answer", "message"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    #用过的prompt，记一下
    question = "You are an expert at finding vulnerability in code\n{code}.\nYou are review the answer from your colleague just made. Following the instrctions below, please JUST select the most probable result from the {answer}.\n" + message
    after_rag_chain = prompt| model_local | parser
    after_rag_chain = after_rag_chain.with_retry()
    answer = after_rag_chain.invoke({"code": code_content, "answer": answer, "message": message})
    print(answer)
    
    after_test_result = r"results/after_rag_result.json"
    with open(after_test_result, "a") as f:
        json.dump(answer, f, indent=2)


def one_detection(func_value, retriever, conf):
    model_local = ChatOllama(model=conf.analysis.model, temperature=conf.analysis.temperature, format=conf.analysis.format, num_ctx=conf.analysis.num_ctx, base_url=conf.analysis.base_url)
    
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
    
    
    pdf_flag = 0
    if pdf_flag == 1:
        page_content = read_pdf()
    else:
        page_content = read_doc()
        
    print("read docucments complete")
    
    retriever = vector_embedding(page_content, conf)
    
    print("retrieve complete...")
    
    with open("testcase/service.c", "r") as f:
        code_content = f.read()
    code_content = remove_comments(code_content)
    one_detection(code_content, retriever, conf)
    
    print("test complete...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="yaml file for config.")
    args = parser.parse_args()
    main(args)