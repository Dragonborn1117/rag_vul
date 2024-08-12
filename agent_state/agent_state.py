from typing import TypedDict, Annotated, List, Tuple, Dict

def add_message(left:list, right:list) -> list:
    return left + right

def replace(left:str, right:str) -> str:
    return right

def do_nothing(left:list, right:list) -> str:
    return left

class AgentState(TypedDict):
    messages: Annotated[list, add_message]

class CodeAgentState(AgentState):
    code: Annotated[List[Tuple[str, str]], add_message]

class ContextAgentState(CodeAgentState):
    pass

class AuditorAgentState(CodeAgentState):
    pass

class JudgeAgentState(CodeAgentState):
    pass

class KnowledgeReaderAgentState(CodeAgentState):
    pass

class StepAdapterAgentState(CodeAgentState):
    pass
