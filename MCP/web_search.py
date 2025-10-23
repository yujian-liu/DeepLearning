import asyncio
import os

import httpx
import requests
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from mcp.server import FastMCP
from zai import ZhipuAiClient

_ = load_dotenv(find_dotenv())

app = FastMCP("web-search")

llm = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0
)

@app.tool()
async def web_search(query: str) -> str:
    """
        搜索互联网内容

        Args:
            query: 要搜索内容

        Returns:
            搜索结果的总结
    """
    url = "https://open.bigmodel.cn/api/paas/v4/web_search"

    payload = {
        "search_query": query,
        "search_engine": "search_pro",
        "search_intent": False,
        "count": 10,
        "search_recency_filter": "noLimit",
        "content_size": "medium",
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('ZHIPU_API_KEY')}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    return response.json()['search_result'][0]['content']

    # client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))
    # message = [{'role': 'user', 'content': query}]
    # response = client.chat.completions.create(
    #     model='glm-4.5',
    #     messages=message,
    #     temperature=0
    # )
    # return response.choices[0].message.content

    # template = "回答我提出的问题：{query}\n\n\n结果中不包含表情"
    # prompt = ChatPromptTemplate.from_template(template)
    # chain = prompt | llm
    # response = await chain.ainvoke(query)
    # return response.content

if __name__ == "__main__":
    # asyncio.run(web_search('今日宁波天气'))
    app.run(transport="stdio")
