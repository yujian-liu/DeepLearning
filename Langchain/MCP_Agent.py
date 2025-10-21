import asyncio
import os
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import BaseMessage  # 新增：导入消息基类

_ = load_dotenv(find_dotenv())

from langchain_community.chat_models import ChatZhipuAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

llm = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("在获得工具的返回后判断是否需要调用其他工具还是直接返回结果\
                                            在最终的回答中仅包含问题的答案,以下是输入内容：{messages}")

async def main():
    # 设置 MCP 客户端
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["./Math_server.py"],  # 确保路径正确
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp/",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()

    # 绑定工具到模型
    model_with_tools = prompt | llm.bind_tools(tools)

    # 创建工具节点
    tool_node = ToolNode(tools)

    # 决定是否继续调用工具
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.type == "ai" and last_message.tool_calls:
            return "tools"
        return END

    # 定义模型调用节点
    async def call_model(state: MessagesState):
        messages = state["messages"]
        # print(messages)  # 调试用：打印当前消息
        response = await model_with_tools.ainvoke(messages)  # 模型生成响应（BaseMessage 类型）
        return {"messages": [response]}  # 注意：直接返回消息对象，而非 response.content

    # 构建图（使用内置的 MessagesState，无需自定义 State）
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    # 定义图的流向
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        should_continue,
    )
    builder.add_edge("tools", "call_model")

    # 编译图
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "a001"}}

    # 测试
    async for event in graph.astream(
            {"messages": ["what's (3 + 5) x 12 + 1 + 3?"]},
            stream_mode="values",
            config=config,
    ):
        event["messages"][-1].pretty_print()

    async for event in graph.astream(
            {"messages": ["what is the weather in nyc?"]},
            stream_mode="values",
            config=config,
    ):
        event["messages"][-1].pretty_print()

    async for event in graph.astream(
            {"messages": ["who is Donald John Trump?"]},
            stream_mode="values",
            config=config,
    ):
        event["messages"][-1].pretty_print()

    # math_response = await graph.ainvoke(
    #     {"messages": ["what's (3 + 5) x 12?"]},
    #     config=config,
    # )
    # print("Math response:", math_response["messages"][-1].content)
    #
    # weather_response = await graph.ainvoke(
    #     {"messages": ["what is the weather in nyc?"]},
    #     config=config,
    # )
    # print("Weather response:", weather_response["messages"][-1].content)

if __name__ == '__main__':
    asyncio.run(main())