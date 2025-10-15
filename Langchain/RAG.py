import os

from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

import bs4
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ = load_dotenv(find_dotenv())

from langchain_community.chat_models import ChatZhipuAI

# llm 模型
llm = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0
)

# embedding模型
model_name = "BAAI/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 向量仓库
vector_store = InMemoryVectorStore(embeddings)

# 加载数据
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 数据切片/分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(all_splits)

# 加载用于rag的prompt
prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    # 检索相关文档
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    # 拼接相关文档
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})

    response = llm.invoke(messages)
    return {"answer": response.content}

# langgraph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])

# 将图像保存为文件
with open("./img/rag_pic.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

print("图像已保存")
