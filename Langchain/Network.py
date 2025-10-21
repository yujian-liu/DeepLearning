import operator
import os
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

class State(TypedDict):
    message: Annotated[str, lambda x, y: y]
    num: Annotated[int, operator.add]

def fc_a(state: State):
    print("a")
    return {"message": state["message"], "num": 1}

def fc_b(state: State):
    print("b")
    return {"message": state["message"], "num": 1}

def fc_c(state: State):
    print("c")
    return {"message": state["message"], "num": 1}

graph_builder = StateGraph(State)
graph_builder.add_node("a", fc_a)
graph_builder.add_node("b", fc_b)
graph_builder.add_node("c", fc_c)

graph_builder.set_entry_point("a")
graph_builder.add_edge("a", "b")
graph_builder.add_edge("a", "c")

graph = graph_builder.compile()
response = graph.invoke({"message": "Hello World", "num": 0})
print(response["message"])
print(response["num"])

# 将图像保存为文件
# with open("./img/network_pic.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())
#
# print("图像已保存")