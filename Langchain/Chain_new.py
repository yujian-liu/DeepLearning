import os
from typing import Literal, Optional, TypedDict

from langgraph.constants import START, END
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from langchain.chains.router import LLMRouterChain, MultiPromptChain, RouterChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph

_ = load_dotenv(find_dotenv())

llm = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)

chain = prompt | llm
product = "Queen Size Sheet Set"
# response = chain.invoke(product)
# print(response.content)

# SequentialChain
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
chain_one = first_prompt | llm

second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company:{company_name}"
)
chain_two = second_prompt | llm

overall_simple_chain = chain_one | chain_two
# response = overall_simple_chain.invoke(product)
# print('-'*10)
# print(response.content)

# Router Chain
#第一个提示适合回答物理问题
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

#第二个提示适合回答数学问题
math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

#第三个适合回答历史问题
history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

#第四个适合回答计算机问题
computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    },
    {
        "name": "History",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": computerscience_template
    }
]

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = prompt | llm
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = default_prompt | llm
destination_chains["DEFAULT"] = default_chain

MULTI_PROMPT_ROUTER_TEMPLATE = """
Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a JSON object with a single key "destination".
The value of "destination" must be one of: {destinations}, or "DEFAULT".
Do NOT return any markdown, code blocks, backticks, or explanations.
Only return the raw JSON object as a string. No wrapping, no formatting.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}
"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = ChatPromptTemplate.from_template(
    template=router_template
)

destination_names = ["DEFAULT"] + [p_info["name"] for p_info in prompt_infos]
destinationType = Literal[tuple(destination_names)]
class RouteQuery(TypedDict):
    destination: destinationType

router_chain = router_prompt | llm.with_structured_output(RouteQuery, method="function_calling")

class State(TypedDict):
    input: str
    destination: destinationType
    answer: str

def route_query(state: State):
    response = router_chain.invoke({"input": state["input"]})
    return {"destination": response["destination"]}

def prompt_factory(name):
    def method(state: State):
        response = destination_chains[name].invoke({"input": state["input"]})
        return {"answer": response.content}
    return method

def select_node(state: State) -> destinationType:
    return state["destination"]

graph = StateGraph(State)
graph.add_node("route_query", route_query)
graph.add_edge(START, "route_query")
graph.add_conditional_edges("route_query", select_node)
for destination in destination_names:
    method = prompt_factory(destination)
    graph.add_node(destination, method)
    graph.add_edge(destination, END)
app = graph.compile()

# response = router_chain.invoke("What is black body radiation")
# print(response)

response = app.invoke({"input": "What is black body radiation"})
print("-"*10)
print(response)
print(type(response))
print(response["answer"])

# 将图像保存为文件
with open("./img/chain_pic.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

print("图像已保存")
