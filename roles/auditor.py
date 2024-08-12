from .__base import Role
from langgraph.graph import Graph
from typing import List, Dict, Tuple, Literal
from agent_state.agent_state import AuditorAgentState
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama

class Auditor(Role):
    def __init__(self, code:List[Tuple[str,str]], name: str = 'auditor'):
        super().__init__(name)
        self.code = code
        self.model = ChatOllama(model = "llama3")
    
    def prompt_render(self, question:str, code:List[Tuple[str,str]]) -> str:
        prompt = "I will give you some code with description and ask you a question about it. Your task is to answer the question based on the code. Here is the code:\n"
        for i, (c, desc) in enumerate(code):
            prompt += f"```{c}```\n{desc}\n\n"
        
        prompt += f"Question: {question}"

        prompt += "When answer the question, please use a statement like \"The code does/does not ...\""

        return prompt

    def ask_question(self, state:AuditorAgentState) -> Dict[str, list]:
        return self.call_model(state)

    def get_graph(self) -> Graph:
        workflow = StateGraph(AuditorAgentState)
        workflow.add_node("ask_question", self.ask_question)
        workflow.add_edge("ask_question", END)

        workflow.set_entry_point("ask_question")
        return workflow

    def run(self, input: str, **kargs) -> str:
        graph = self.get_graph()
        runnable = graph.compile()
        result:AuditorAgentState = runnable.invoke({
            "messages": [HumanMessage(
                self.prompt_render(input, self.code),
            )],
            "code": self.code
        })

        assert isinstance(result["messages"][-1], AIMessage)
        return result["messages"][-1].content
