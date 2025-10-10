# https://python.langchain.com/v0.2/docs/tutorials/chatbot/
import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

_ = load_dotenv(find_dotenv())

from langchain_community.chat_models import ChatZhipuAI

model = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0
)

# no RunnableWithMessageHistory
# response = model.invoke([HumanMessage(content="Hi! I'm Bob")])
# print(response.content)
# print('----------')
# response = model.invoke([HumanMessage(content="What's my name")])
# print(response.content)

# response = model.invoke(
#     [
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hi Bob! 👋 Nice to meet you! How can I help you today? Feel free to ask me anything or let me know what you're working on."),
#         HumanMessage(content="What's my name")
#     ]
# )
# print(response.content)

# with RunnableWithMessageHistory
# 存储对话记录
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# with_message_history = RunnableWithMessageHistory(model, get_session_history)

# 不同session_id代表不同会话
# config = {'configurable': {'session_id': 'abc2'}}
#
# response = with_message_history.invoke(
#     [HumanMessage(content="Hi! I'm Bob")],
#     config=config
# )
# print(response.content)
# print('-'*10)
#
# response = with_message_history.invoke(
#     [HumanMessage(content="What's my name?")],
#     config=config
# )
# print(response.content)
# print('-'*10)
#
# config = {'configurable': {'session_id': 'abc3'}}
# response = with_message_history.invoke(
#     [HumanMessage(content="What's my name?")],
#     config=config
# )
# print(response.content)

# add prompt
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant. Answer all questions to the best of your ability."
#         ),
#         MessagesPlaceholder(variable_name="messages")
#     ]
# )
#
# chain = prompt | model

# response = chain.invoke({"messages": [HumanMessage(content="hi! I'm Bob")]})
# print(response.content)
# print("-"*10)

# with_message_history = RunnableWithMessageHistory(model, get_session_history)
# config = {"configurable": {"session_id": "abc5"}}
# response = with_message_history.invoke(
#     [HumanMessage(content="Hi! I'm Jim")],
#     config=config
# )
# print(response.content)
# print('-'*10)
#
# response = with_message_history.invoke(
#     [HumanMessage(content="What's my name?")],
#     config=config
# )
# print(response.content)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

# response = chain.invoke(
#     {"messages": [HumanMessage(content="Hi! I'm Bob")], "language": "Spanish"}
# )
#
# print(response.content)

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

# config = {"configurable": {"session_id": "abc11"}}
#
# response = with_message_history.invoke(
#     {"messages": [HumanMessage(content="Hi! I'm Todd")], "language": "Spanish"},
#     config,
# )
#
# print(response.content)
#
# response = with_message_history.invoke(
#     {"messages": [HumanMessage(content="What's my name")], "language": "Spanish"},
#     config,
# )
# print(response.content)

# stream
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm Todd, tell me a joke")],
        "language": "English",
    },
    config
):
    print(r.content, end='')

# 对Memory的操作（ConversationBufferWindowMemory等）需要另外手动处理
