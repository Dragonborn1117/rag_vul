from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, HumanMessage, BaseMessage
from langchain_community.chat_models import ChatOllama
from .__base import Role
from typing import Literal, Dict, List, Optional, get_args, Tuple
from agent_state.agent_state import ContextAgentState
from langchain_core.pydantic_v1 import BaseModel, Field
from loguru import logger
import traceback
from tree_sitter import Language, Parser
import tree_sitter_c as tsc


# available_context_tools = Literal["GetFunctionsList", "GetStateVariablesList", "GetStateVariablesDetail", "GetFunctionCodeAST"]
available_context_tools = Literal["GetFunctionsList"]

tools_description = {
    "GetFunctionsList": "Get the list of functions in code",
    # "GetStateVariablesList": "Get the list of state variables in code",
    # "GetStateVariablesDetail": "Get the detail of a specific state variable",
    # "GetFunctionCode": "Get the ast of a function in code"
}

class statistic_tools():
    def __init__(self, context: str="code"):
        self.cpp_code_snippet = context
    
    def query_context(self, cpp_query_text: str="query"):
        result = []
        
        C_LANGUAGE = Language(tsc.language())
        parser = Parser(C_LANGUAGE) 
        
        # 定义query
        query = C_LANGUAGE.query(cpp_query_text)

        # 获取具体语法树
        self.tree = parser.parse(bytes(self.cpp_code_snippet, "utf8"))
        root_node = self.tree.root_node

        # 获取节点
        # capture: list[Node, str]
        capture = query.captures(root_node)
        for node, _ in capture:
            result.append(node.text)
            print(f"{node.text}")
        
        return result
    
    def getFunctionsList(self):
        cpp_query_text = '''
        (function_declarator declarator: (identifier)@1 )
        '''
        self.query_context(cpp_query_text)
    
    def getStateVariablesList(self):
        cpp_query_text = '''
        (function_declarator declarator: (identifier)@1 )
        (initializer_list) @2
        ( call_expression ) @3
        (assignment_expression  right:(_) @4)
        '''
        self.query_context(cpp_query_text)
    
    def getStateVariablesDetail(self):
        cpp_query_text = '''
        (assignment_expression  right:(_) @1)
        '''
        self.query_context(cpp_query_text)
    
    def getFunctionCode(self):
        cpp_query_text = '''
        ( call_expression ) @1
        '''
        self.query_context(cpp_query_text)

class ContextReader(Role):

    need_external_prompt = "To answer this question base on the given code, do you need extra code?"
    
    def __init__(self, code: str="code to detect", name: str="ContextReader"):
        super().__init__(name)
        self.code = code
        self.model = ChatOllama(model="llama3")
        self.previous_structured_original = None

    def need_extra(self, state: ContextAgentState) -> Dict[str, list]:
        state["messages"].append(self.need_external_prompt)
        return self.call_model(state, self.YesNoQuestionStructure)
    
    def need_extra_state_check(self, state: ContextAgentState) -> Literal["choose_tools", "__end__"]:
        assert isinstance(self.previous_structured_original, self.YesNoQuestionStructure)
        if self.previous_structured_original.answer == True:
            return "choose_tools"
        else:
            return "__end__"

    def choose_tool_state_check(self, state: ContextAgentState) -> available_context_tools:
        assert isinstance(self.previous_structured_original, self.ChooseToolStructure)
        return self.previous_structured_original.tool
    
    def choose_tool(self, state: ContextAgentState) -> List[BaseMessage]:
        description = "\n".join([f"{tool}: {description}" for tool, description in tools_description.items()])
        state["messages"].append(f"""You can use the following tools to get the information you need: {', '.join(get_args(available_context_tools))}.

{description}

Which tool would you like to use?""")

        response = self.call_model(state, self.ChooseToolStructure)
        assert isinstance(self.previous_structured_original, self.ChooseToolStructure)
        assert self.previous_structured_original.tool in get_args(available_context_tools)

        logger.info("AI selected tool: {}", self.previous_structured_original.tool)

        return response
    
    def GetFunctionsList(self, state: ContextAgentState) -> List[BaseMessage]:
        # 第一步，询问LLM，需要获取哪个合约的函数列表
        logger.info("Getting functions list")
        
        response = self.call_model(state, self.ChooseCodeNameForFuncListStructure)
        assert isinstance(self.previous_structured_original, self.ChooseCodeNameForFuncListStructure)
        code_name = self.previous_structured_original.code_name
        try:
            logger.info("Returning functions list")
            # 111
            result_list = statistic_tools().getFunctionsList()
            return {"messages": [HumanMessage(result_list)]}
        
        except:
            logger.info("To LLM >> Function list not found. You can use other tools or trg again.")
            return {"messages": [HumanMessage("Function list not found. You can use other tools or try again")]}
    
    # def GetStateVariablesList(self, state: ContextAgentState) -> List[BaseMessage]:

    #     logger.info("Getting state variables list")
    #     assert isinstance(self.previous_structured_original, self.ChooseCodeNameForStateVarListStructure)
    #     code_name = self.previous_structured_original.code_name
        
    #     try:
    #         logger.info("Returning state variables list")
    #         state["code"].append(("\n".join([x.name for x in contract.state_variables]), "State variables in " + contract_name))
    #         return {"messages": [HumanMessage("\n".join([x.name for x in contract.state_variables_declared]))]}
    #     except:
    #         logger.info("To LLM >> State Variable list not found. You can use other tools or try again.")
    #         return {"messages": [HumanMessage("State Variable list not found. You can use other tools or try again.")]}
    
    # def GetStateVariableDetail(self, state: ContextAgentState) -> List[BaseMessage]:
    #     # 第一步，询问LLM，需要获取哪个合约的状态变量的详细信息
    #     logger.info("Getting state variable detail")
        
    #     response = self.call_model(state, self.ChooseCodeNameForStateVarDetailStructure)
    #     assert isinstance(self.previous_structured_original, self.ChooseCodeNameForStateVarDetailStructure)
    #     code_name = self.previous_structured_original.code
    #     try:
    #         state_variable_name = self.previous_structured_original.state_variable_name
    #         state_variable = contract.get_state_variable_from_name(state_variable_name)

    #         expression = (state_variable.type.name if isinstance(state_variable.type, falcon.core.solidity_types.type.Type) else " ".join([x.name for x in state_variable.type]) )+ " " + state_variable.visibility + " " + state_variable.name
    #         return {"messages": [HumanMessage(expression)]}
    #     except:
    #         logger.info("To LLM >> State variable detail not found. You can use other tools or try again.")
    #         return {"messages": [HumanMessage("State variable detail not found. You can use other tools or try again.")]}

    # def GetFunctionCodeAST(self, state: ContextAgentState) -> List[BaseMessage]:
    #     # 第一步，询问LLM，需要获取哪个合约的函数的详细信息
    #     logger.info("Getting function detail")
        
    #     assert isinstance(self.previous_structured_original, self.ChooseCodeNameForFuncDetailStructure)
    #     code_name = self.previous_structured_original.code_name
    #     try:
    #         # 判断function_name是否存在
    #         function_name = self.previous_structured_original.function_name
    #         functions = list(filter(lambda x: x.name == function_name, contract.functions))
    #         if len(functions) == 0:
    #             logger.info("To LLM >> Function not found. You can use the GetFunctionsList tool to get the list of functions.")
    #             return {"messages": [HumanMessage("Function not found. You can use the GetFunctionsList tool to get the list of functions.")]}
    #         # 如果存在，返回源码中的详细的定义
    #         function = functions[0]
    #         filename = function.source_mapping.filename
    #         start_pos = function.source_mapping.start
    #         expression_len = function.source_mapping.length
    #         with open(filename, "rb") as f:
    #             f.seek(start_pos)
    #             expression = f.read(expression_len).decode("utf-8")
    #         logger.info("Returning function detail")
    #         state["code"].append((expression, "Function detail in " + contract_name))
    #         return {"messages": [HumanMessage(expression)]}
    #     except:
    #         logger.info("To LLM >> Function ast not found. You can use other tools.")
    #         return {"messages": [HumanMessage("Function ast not found. You can use other tools.")]}

    class YesNoQuestionStructure(BaseModel):
        answer: bool = Field(description="The answer to the yes/no question. True for yes, False for no")

        def __str__(self) -> str:
            return "Yes" if self.answer else "No"
    
    class ChooseToolStructure(BaseModel):
        tool: str = Field(description="The name of the tool to use")

        def __str__(self) -> str:
            return "I would like to use the tool: " + self.tool
        
    class ChooseCodeNameForFuncListStructure(BaseModel):
        code_name: str = Field(description="The name of the code to get the functions list for")

        def __str__(self) -> str:
            return "I would like to get the functions list for the code: " + self.code_name
    
    class ChooseCodeNameForStateVarListStructure(BaseModel):
        code_name: str = Field(description="The name of the code to get the state variable list for")

        def __str__(self) -> str:
            return "I would like to get the state variable list for the code: " + self.code_name
        
    class ChooseCodeNameForStateVarDetailStructure(BaseModel):
        code_name: str = Field(description="The name of the code to get the state variable detail for")
        state_variable_name: str = Field(description="The name of the state variable to get the detail for")

        def __str__(self) -> str:
            return "I would like to get the state variable detail for the code: " + self.code_name + " and the state variable: " + self.state_variable_name
        
    class ChooseContractNameForFuncDetailStructure(BaseModel):
        code_name: str = Field(description="The name of the code to get the function detail for")
        function_name: str = Field(description="The name of the function to get the detail for")

        def __str__(self) -> str:
            return "I would like to get the function detail for the contract: " + self.contract_name + " and the function: " + self.function_name

    def get_graph(self) -> Graph:

        workflow = StateGraph(ContextAgentState)
        workflow.add_node("need_extra", self.need_extra)
        
        workflow.add_node("choose_tools", self.choose_tool)
        
        workflow.add_conditional_edges("need_extra", self.need_extra_state_check)

        workflow.add_conditional_edges("choose_tools", self.choose_tool_state_check)

        workflow.add_node("GetFunctionsList", self.GetFunctionsList)
        
        # workflow.add_node("GetStateVariablesList", self.GetStateVariablesList)

        # workflow.add_node("GetStateVariableDetail", self.GetStateVariableDetail)
 
        # workflow.add_node("GetFunctionCode", self.GetFunctionCodeAST)

        workflow.add_edge("GetFunctionsList", "need_extra")
        # workflow.add_edge("GetStateVariablesList", "need_extra")
        # workflow.add_edge("GetStateVariableDetail", "need_extra")
        # workflow.add_edge("GetFunctionCodeAST", "need_extra")

        workflow.set_entry_point("need_extra")

        return workflow
    @logger.catch
    def run(self, input_text: str, input_code: str, code_language: str=None, current_function: str=None) -> List[Tuple[str,str]]:
        graph = self.get_graph()
        runnable = graph.compile()
        while True:
            try:
                result:ContextAgentState = runnable.invoke({
                    "messages": [HumanMessage(content = input_text + "\n\n" + input_code + "\n\nCurrent Code is: " + code_language + "\nCurrent Function is: " + current_function)],
                    "code": [(input_code, f"code language: {code_language}, function: {current_function}")]
                })
                break
            except:
                logger.error(traceback.format_exc())

        cleaned_result = self.code_clean_based_on_description(result["code"])

        return cleaned_result

    def code_clean_based_on_description(self, code:List[Tuple[str, str]])-> List[Tuple[str,str]]:
        # 如果code的第二项重复，删除重复的
        code_dict = {}
        for item in code:
            if item[1] not in code_dict:
                code_dict[item[1]] = item[0]
    
        return [(v, k) for k, v in code_dict.items()]

