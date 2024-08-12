from abc import ABC, abstractmethod
from registry import register_role
from langgraph.graph import Graph
from typing import List, Dict, Tuple, Literal, Optional
from agent_state.agent_state import AgentState
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_community.chat_models import ChatOllama

class Role:
    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        register_role(cls)
    
    def __init__(self, name: str):
        self.name = name
        self.model: ChatOllama = None
        self.previous_structured_original = None

    def call_model(self, state: AgentState, structure:Optional[BaseModel]=None) -> Dict[str, list]:
        messages = state["messages"]
        if structure == None:
            response = self.model.invoke(messages)
            return {"messages": [response]}
        else:
            response = self.model.with_structured_output(structure).invoke(messages)
            self.previous_structured_original = response
            return {"messages": [AIMessage(str(response))]}
    
    @abstractmethod
    def get_graph(self) -> Graph:
        pass

    @abstractmethod
    def run(self, input: str, **kargs) -> Graph:
        pass
