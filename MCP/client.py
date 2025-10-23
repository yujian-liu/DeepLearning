import asyncio

from mcp import StdioServerParameters, stdio_client, ClientSession

server_params = StdioServerParameters(
    command="python",
    args=["web_search.py"],
)

async def main():
    # 创建 stdio 客户端
    async with stdio_client(server_params) as (stdio, write):
        # 创建 ClientSession 对象
        async with ClientSession(stdio, write) as session:
            # 初始化 ClientSession
            await session.initialize()

            response = await session.list_tools()
            print(response)

            response = await session.call_tool('web_search', {'query': '1+1的结果是什么?'})
            print(response)

if __name__ == '__main__':
    asyncio.run(main())