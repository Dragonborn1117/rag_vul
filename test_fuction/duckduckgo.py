from langchain_community.tools import DuckDuckGoSearchRun
import os

os.environ["http_proxy"] = "http://10.201.149.18:7897" 
os.environ["https_proxy"] = "https://10.201.149.18:7897" 


search = DuckDuckGoSearchRun()

search.invoke("Obama's first name?")
