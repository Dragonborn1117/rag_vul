from .__base import Role
from langgraph.graph import Graph
from typing import List, Dict, Tuple, Literal
from agent_state.agent_state import JudgeAgentState
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field


class Judge(Role):
    def __init__(self, code:str, summarizations:List[str] , name: str = "judge"):
        super().__init__(name)

        self.code = code
        self.summarizations = summarizations

        self.model = ChatOllama(model = "llama3")

    def prompt_render(self, knowledge:str, code:str, summarizations:List[str]) -> str:
        prompt = "You are a code auditing expert. I will give you a code and some summarizations from other experts. Your task is to answer whether a specific type of vulnerability exists in the code and summarizations. Here is the code:\n"
        prompt += f"```{code}```\n"
        prompt += "Summarizations:\n"
        for i, s in enumerate(summarizations):
            prompt += f"{i+1}. {s}\n"

        prompt += f"Vulnerability: {knowledge}"

        return prompt
    
    def ask_question(self, state:JudgeAgentState) -> Dict[str, list]:
        return self.call_model(state, structure=self.JudgeResult)
    
    class JudgeResult(BaseModel):
        has_vuln: bool = Field(description="Whether the code has the given type of vulnerability")
        reason: str = Field(description="The reason for the result")

        def __str__(self):
            return f"Has vulnerability: {self.has_vuln}\nReason: {self.reason}"

    def get_graph(self) -> Graph:
        workflow = StateGraph(JudgeAgentState)
        workflow.add_node("ask_question", self.ask_question)
        workflow.add_edge("ask_question", END)

        workflow.set_entry_point("ask_question")
        return workflow
    
    def run(self, input: str, **kargs) -> dict:
        graph = self.get_graph()
        runnable = graph.compile()
        result:JudgeAgentState = runnable.invoke({
            "messages": [HumanMessage(
                self.prompt_render(input, self.code, self.summarizations),
            )],
            "code": self.code
        })

        assert isinstance(self.previous_structured_original, self.JudgeResult)
        return {"has_vuln": self.previous_structured_original.has_vuln, "reason": self.previous_structured_original.reason}
