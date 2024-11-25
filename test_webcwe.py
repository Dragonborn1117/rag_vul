import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from fake_useragent import UserAgent
from langchain.schema import Document
import os
import re

os.environ['USER_AGENT'] = UserAgent().chrome

def clean_text(raw_text):
    # 去掉多余的换行符和空格
    clean = re.sub(r'\s+', ' ', raw_text)
    return clean.strip()

def add_mark_text(text):
    words_to_add_hash = ["Description", "Extended Description", "Alternate Terms", "Demonstrative Examples"]
    pattern = r"\b(" + "|".join(re.escape(word) for word in words_to_add_hash) + r")\b"
    result = re.sub(pattern, r"\n# \1 \n", text)
    return result
    

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(id=("Description", "Extended_Description", "Alternate_Terms", "Demonstrative_Examples"))
loader = WebBaseLoader(
    web_paths=("https://cwe.mitre.org/data/definitions/119.html",),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()
cleaned_content = ""
for doc in docs:
    cleaned_content = clean_text(doc.page_content)
    cleaned_content = add_mark_text(cleaned_content)
    print(cleaned_content)


llm = ChatOllama(model="llama3.1", temperature=0, format="json", num_ctx=8192, base_url="http://localhost:8080")
headers_to_split_on = [
    ("#", "Header 1")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = markdown_splitter.split_text(cleaned_content)
for i, chunk in enumerate(splits, 1):
    print(f"块 {i}:\n{chunk.page_content}\n{'-' * 50}")

vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(base_url="http://localhost:8080", model="bge-m3"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("What is CWE-119")
print(retrieved_docs[0].page_content)
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What is CWE-119"))