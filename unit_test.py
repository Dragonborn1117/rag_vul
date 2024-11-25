from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_community.document_loaders import WebBaseLoader
import signal
import argparse
from omegaconf import OmegaConf
import json
import re
import os
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from fake_useragent import UserAgent
import bs4

class Jsonoutput(BaseModel):
    Vulnerability_Detection: str = Field(description="If this code snippet has vulnerabilities, output Yes; otherwise, output No.")
    Vulnerability_Assessment: list = Field(description="a list of ONLY THREE results optimized to retrieve the most relevant results of the code snippet, and give more star numbers if the answer is more probable.") 
    Vulnerability_Location: str = Field(description="Provide a vulnerability location result for the vulnerable code snippet.") 

class Argueoutput(BaseModel):
    Vulnerability_Detection: str = Field(description="If this code snippet has vulnerabilities, output Yes; otherwise, output No.")
    # select_answer: str = Field(description="select the most probable answer from the Vulnerability_Assessment")
    select_answer: str = Field(description="whether a specific type of vulnerability exists in the code and summarizations")
    
class vul():
    def __init__(self, args):
        if args.config == None:
            args.config = "configs/base.yaml"
            
        self.conf = OmegaConf.load(args.config)
        self.retriever = None
    
    def timeout_handler(signum, frame):
        raise TimeoutError('Model doesn\'t response for a while') 

    def remove_comments(self, text):
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

    def clean_text(self, raw_text):
        # 去掉多余的换行符和空格
        clean = re.sub(r'\s+', ' ', raw_text)
        return clean.strip()

    def add_mark_text(self, text):
        words_to_add_hash = ["Description", "Extended Description", "Alternate Terms", "Demonstrative Examples"]
        pattern = r"\b(" + "|".join(re.escape(word) for word in words_to_add_hash) + r")\b"
        result = re.sub(pattern, r"\n# \1 \n", text)
        return result

    def vector_embedding(self):
        os.environ['USER_AGENT'] = UserAgent().chrome
        bs4_strainer = bs4.SoupStrainer(id=("Description", "Extended_Description", "Alternate_Terms", "Demonstrative_Examples"))
        loader = WebBaseLoader(
            web_paths=("https://cwe.mitre.org/data/definitions/119.html",),
            bs_kwargs={"parse_only": bs4_strainer},
        )

        docs = loader.load()
        cleaned_content = ""
        for doc in docs:
            cleaned_content = self.clean_text(doc.page_content)
            cleaned_content = self.add_mark_text(cleaned_content)
            print(cleaned_content)
        
        
        headers_to_split_on = [
            ("#", "Header 1")
        ]
            
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits = markdown_splitter.split_text(cleaned_content)
        for i, chunk in enumerate(splits, 1):
            print(f"块 {i}:\n{chunk.page_content}\n{'-' * 50}")

        vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(base_url="http://localhost:8080", model="bge-m3"))
        
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    def with_rag(self, model_local, code_content):
        parser = JsonOutputParser(pydantic_object=Jsonoutput)
        
        # """You are an expert at finding vulnerability in code. \
        #             Given a question, return a list of ONLY THREE results optimized to retrieve the most relevant results. Respond with json.\
        #             If there are acronyms or words you are not familiar with, do not try to rephrase them.
        #             Answer the question based on the following context:{context} 
        #             Question:{question}\n{format_instructions}\n{question}\n.Think step by step""",
                    
        prompt = PromptTemplate(
            template="""You are an expert at finding vulnerability in code. \
                    Given a question, return a list of ONLY THREE results optimized to retrieve the most relevant results. Respond with json.\
                    If there are acronyms or words you are not familiar with, do not try to rephrase them.
                    Answer the question based on the following context:{context} 
                    Question:{question}\n{format_instructions}\n{question}\n.Think step by step""",
            input_variables=["question"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        setup_and_retrieval = RunnableParallel(
            {
                "context": self.retriever,
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

    def one_detection(self, func_value):
        model_local = ChatOllama(model=self.conf.analysis.model, temperature=self.conf.analysis.temperature, format=self.conf.analysis.format, num_ctx=self.conf.analysis.num_ctx, base_url=self.conf.analysis.base_url)
        
        code_content = self.remove_comments(func_value)

        try:
            signal.alarm(100)
            self.with_rag(model_local, code_content)
            
        except Exception as e:
            signal.alarm(0)
            print(e)
            
    def main(self):
        
        signal.signal(signal.SIGALRM, self.timeout_handler)

        self.vector_embedding()
        
        print("retrieve complete...")
        
        with open("testcase/target0.c", "r") as f:
            code_content = f.read()
        code_content = self.remove_comments(code_content)
        
        self.one_detection(code_content)
        
        print("test complete...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="yaml file for config.")
    args = parser.parse_args()
    vul(args).main()
    