import os
from operator import add
from typing import TypedDict, List, Literal, Annotated

from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command

_ = load_dotenv(find_dotenv())

from langchain_community.chat_models import ChatZhipuAI

# llm 模型
llm = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0
)

narrator_prompt_template = """
    这是一个关于正派与反派最终决战的故事，以正派与反派的对话为主，你是一个旁白，你需要描写决战的环境以及必要的旁白内容，\
    当前剧情：
    {context}

    请从当前剧情开始续写，直到故事应当结束或者后续内容应由正派或者反派进行描写\
    以严格的 JSON 格式返回，只包含 next_agent 和 content 两个字段，不要任何额外说明或 Markdown 代码块。\
    next_agent只能是hero、villain、finish这三个的其中一个，\
    其中hero表示接下来应该由正派来进行描写，villain表示应当由反派描写，finish表示故事应当结束,\
    content是你续写的内容
    
    只返回 JSON 对象，例如：{{"next_agent": "hero", "content": "..."}}\
"""

hero_prompt_template = """
    这是一个关于正派与反派最终决战的故事，以正派与反派的对话为主，你是一个正派，你需要描写正派的行动、神态、语言、内心活动等，\
    当前剧情：
    {context}

    请从当前剧情开始续写，直到故事应当结束或者后续内容应由旁白或者反派进行描写\
    以严格的 JSON 格式返回，只包含 next_agent 和 content 两个字段，不要任何额外说明或 Markdown 代码块。\
    next_agent只能是narrator、villain、finish这三个的其中一个，\
    其中narrator表示接下来应该由旁白来进行描写，villain表示应当由反派描写，finish表示故事应当结束,\
    content是你续写的内容
    
    只返回 JSON 对象，例如：{{"next_agent": "narrator", "content": "..."}}\
"""

villain_prompt_template = """
    这是一个关于正派与反派最终决战的武侠故事，以正派与反派的对话为主,包含多轮对话，你是一个反派，你需要描写反派的行动、神态、语言、内心活动等，\
    当前剧情：
    {context}

    请从当前剧情开始续写，直到故事应当结束或者后续内容应由旁白或者正派进行描写\
    以严格的 JSON 格式返回，只包含 next_agent 和 content 两个字段，不要任何额外说明或 Markdown 代码块。\
    next_agent只能是narrator、hero、finish这三个的其中一个，\
    其中narrator表示接下来应该由旁白来进行描写，hero表示应当由正派描写，finish表示故事应当结束,\
    content是你续写的内容
    
    只返回 JSON 对象，例如：{{"next_agent": "hero", "content": "..."}}\
"""

narrator_prompt = ChatPromptTemplate.from_template(narrator_prompt_template)
hero_prompt = ChatPromptTemplate.from_template(hero_prompt_template)
villain_prompt = ChatPromptTemplate.from_template(villain_prompt_template)

class responseType(TypedDict):
    next_agent:str
    content:str

narrator_model = narrator_prompt | llm | JsonOutputParser()
hero_model = hero_prompt | llm | JsonOutputParser()
villain_model = villain_prompt | llm | JsonOutputParser()

class State(TypedDict):
    story: Annotated[List[str], add]

# 旁白
def narrator(state: State) -> Command[Literal["hero", "villain", "finish"]]:
    response = narrator_model.invoke({"context": "\n".join(state["story"])})

    return Command(
        goto=response["next_agent"],
        update={"story": ["旁白：\n" + response["content"]]},
    )

# 主角
def hero(state: State) -> Command[Literal["narrator", "villain", "finish"]]:
    response = hero_model.invoke({"context": "\n".join(state["story"])})

    return Command(
        goto=response["next_agent"],
        update={"story": ["正派：\n" + response["content"]]},
    )
# 反派
def villain(state: State) -> Command[Literal["narrator", "hero", "finish"]]:
    response = villain_model.invoke({"context": "\n".join(state["story"])})

    return Command(
        goto=response["next_agent"],
        update={"story": ["反派：\n" + response["content"]]},
    )

def finish(state: State) -> Command[END]:
    prompt_template = """
        这是一段剧本，根据剧本的内容用金庸的语言风格生成一段小说内容，以下是局部内容：
        {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    finish_model = prompt | llm
    response = finish_model.invoke({"context": "\n".join(state["story"])})
    return Command(
        goto=END,
        update={"story": ["完整故事: \n" + response.content]},
    )

graph_builder = StateGraph(State)
graph_builder.add_node(narrator)
graph_builder.add_node(hero)
graph_builder.add_node(villain)
graph_builder.add_node(finish)

graph_builder.add_edge(START, "narrator")
network = graph_builder.compile()

start_plot = "决战即将开始"
inputs = {"story": [start_plot]}

for chunk in network.stream(inputs, stream_mode="values"):
    if "story" in chunk and len(chunk["story"]) > 0:
        latest = chunk["story"][-1]
        print(f"{latest}\n")

# 将图像保存为文件
# with open("./img/story_maker_pic.png", "wb") as f:
#     f.write(network.get_graph().draw_mermaid_png())
#
# print("图像已保存")