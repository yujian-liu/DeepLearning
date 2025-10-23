import asyncio
import json
import os
import traceback
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from mcp import ClientSession, StdioServerParameters, stdio_client
from zai import ZhipuAiClient

_ = load_dotenv(find_dotenv())

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        # 异步栈,管理异步资源
        self.exit_stack = AsyncExitStack()
        self.client = ZhipuAiClient(api_key=os.environ['ZHIPU_API_KEY'])

    # 初始化session
    async def connect_to_server(self):
        server_params = StdioServerParameters(
            command='python',
            args=['web_search.py'],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )

        await self.session.initialize()

    # 调用MCP服务器与LLM交互
    async def process_query(self, query: str) -> str:
        system_prompt = (
            "You are a helpful assistant."
            "You have the function of online search. "
            "Please MUST call web_search tool to search the Internet content before answering."
            "Please do not lose the user's question information when searching,"
            "and try to maintain the completeness of the question content as much as possible."
            "When there is a date related question in the user's question,"
            "please use the search function directly to search and PROHIBIT inserting specific time."
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query},
        ]

        # 获取mcp服务器tool list
        response = await self.session.list_tools()
        # 生成描述信息
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
        } for tool in response.tools]

        # 将tools信息和query传入LLM
        response = self.client.chat.completions.create(
            model="glm-4.5",
            messages=messages,
            tools=available_tools,
        )

        # 处理返回内容
        content = response.choices[0]
        # LLM选择调用tool
        if content.finish_reason == "tool_calls":
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # 调用对应工具
            result = await self.session.call_tool(tool_name, tool_args)
            print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")

            # 将工具调用记录和工具返回结果补充到messages
            messages.append(content.message.model_dump())
            messages.append({
                'role': 'tool',
                'content': result.content[0].text,
                "tool_call_id": tool_call.id,
            })

            # 将补充后的信息传回LLM，得到最终结果
            response = self.client.chat.completions.create(
                model="glm-4.5",
                messages=messages,
            )
            return response.choices[0].message.content

        # 不需要调用工具
        return content.message.content

    async def chat_loop(self):
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "exit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                traceback.print_exc()

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())