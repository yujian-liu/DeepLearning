# https://python.langchain.com/docs/tutorials/chatbot/
import os
from typing import Sequence

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, add_messages
from typing_extensions import Annotated, TypedDict

_ = load_dotenv(find_dotenv())

from langchain_community.chat_models import ChatZhipuAI

model = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0
)

# workflow = StateGraph(state_schema=MessagesState)
#
# def call_model(state: MessagesState):
#     response = model.invoke(state["messages"])
#     return {"messages": response}
#
# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)
#
# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)
#
# config = {"configurable": {"thread_id": "abc123"}}
# query = "Hi! I'm Bob."
#
# input_messages = [HumanMessage(query)]
# output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

# use prompt
# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You talk like a pirate. Answer all questions to the best of your ability.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )
#
# workflow = StateGraph(state_schema=MessagesState)
#
# def call_model(state: MessagesState):
#     prompt = prompt_template.invoke(state)
#     response = model.invoke(prompt)
#     return {"messages": response}
#
# workflow.add_edge(START, "model")
# workflow.add_node("model", call_model)
#
# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=State)

def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc456"}}
query = "Hi! I'm Bob."
language = "Spanish"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

# stream
config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="")