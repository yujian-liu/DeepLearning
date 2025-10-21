import random
from operator import add
from typing import TypedDict, Literal, List, Annotated

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

class State(TypedDict):
    message: Annotated[List[str], add]

def fc_super(state: State) -> Command[Literal["a", "b", END]]:
    print("super")
    # 在supervisor中的LLM添加辅助判断的tool，则为Supervisor（tool-calling）架构
    return Command(goto=random.choice(["a", "b", END]))

def fc_a(state: State) -> Command[Literal["super"]]:
    print("a")
    return Command(goto="super", update={"message": ["a"]})

def fc_b(state: State) -> Command[Literal["super"]]:
    print("b")
    return Command(goto="super", update={"message": ["b"]})

graph_builder = StateGraph(State)
graph_builder.add_node("super", fc_super)
graph_builder.add_node("a", fc_a)
graph_builder.add_node("b", fc_b)

graph_builder.add_edge(START, "super")

graph = graph_builder.compile()
response = graph.invoke({"message": "test"})
print(response["message"])

# 将图像保存为文件
# with open("./img/supervisor_pic.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())
#
# print("图像已保存")