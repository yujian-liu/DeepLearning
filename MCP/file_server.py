from mcp import SamplingMessage
from mcp.server import FastMCP
from mcp.types import TextContent

app = FastMCP('file_server')

@app.tool()
async def delete_file(file_path: str):
    result = await app.get_context().session.create_message(
        messages=[
            SamplingMessage(
                role='user',
                content=TextContent(type='text', text=f'是否要删除文件：{file_path} (Y)')
            )
        ],
        max_tokens=100
    )

    if result.content.text == 'Y':
        return f'文件{file_path}已被删除！！'

if __name__ == '__main__':
    app.run(transport='stdio')