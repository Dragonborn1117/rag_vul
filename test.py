from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
import signal
import argparse
from omegaconf import OmegaConf



def timeout_handler(signum, frame):
    raise TimeoutError('Model doesn\'t response for a while')

def main(args):
    if args.config == None:
        args.config = "configs/base.yaml"
    
    conf = OmegaConf.load(args.config)
    
    signal.signal(signal.SIGALRM, timeout_handler)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    model_local = ChatOllama(model=conf.analysis.model, temperature=conf.analysis.temperature, format=conf.analysis.format, num_ctx=conf.analysis.num_ctx)

    content_path= r"test/doc/0.md"
    with open(content_path, "r") as f:
        page_content = f.read()
        
    code_path = r"service.c"
    with open(code_path, "r") as f:
        code_content = f.read()

    markdown_document = page_content
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    for i, doc in enumerate(md_header_splits):
        print("-------------------------------------------------------")
        print(f"Document {i+1}:")
        print("Page content:")
        print(doc.page_content)
        print("Metadata:")
        for key, value in doc.metadata.items():
            print(f"{key}: {value}")
        print("\n")

    vectorstore = Chroma.from_documents(
        documents=md_header_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model=conf.analysis.model),
    )

    retriever = vectorstore.as_retriever()

    try:
        signal.alarm(100)
        # print("Before RAG\n")
        # before_rag_template = "What type of {topic} exists in the following code? Respond with json.\n{code}" 
        # before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)
        # before_rag_chain = before_rag_prompt | model_local | JsonOutputParser()
        # print(before_rag_chain.invoke({"topic" : "vulnerability", "code": code_content}))
        
        print("After RAG\n")
        question = "What type of vulnerability exists in the following code? Respond with json.\n" + code_content
        after_rag_template = "Answer the question based on the following context:{context} Question:{question}"
        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        after_rag_chain = {"context" : retriever, "question" : RunnablePassthrough()} | after_rag_prompt | model_local | JsonOutputParser()
        print(after_rag_chain.invoke(question))
    
    except Exception as e:
        signal.alarm(0)
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help="yaml file for config.")
    args = parser.parse_args()
    main(args)