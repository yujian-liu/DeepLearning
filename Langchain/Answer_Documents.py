import os

from dotenv import load_dotenv, find_dotenv
from langchain.indexes import VectorstoreIndexCreator

_ = load_dotenv(find_dotenv())

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_models import ChatZhipuAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.language_models import LLM
from zhipuai import ZhipuAI

# 使用zhupuai自定义LLM类
class ZhipuAILLM(LLM):
    # 显式声明参数（避免动态赋值导致 Pydantic 报错）
    api_key: str
    model: str = "glm-4"
    temperature: float = 0.7

    def __init__(self, api_key: str, model: str = "glm-4", temperature: float = 0.7, **kwargs):
        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return "zhipuai"

    def _call(self, prompt: str, **kwargs) -> str:
        try:
            client = ZhipuAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=512
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print("❌ ZhipuAI 调用错误:", e)
            return f"Error: {e}"

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file, encoding="utf-8")

# 方法一：VectorstoreIndexCreator().query()
model_name = "BAAI/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."
llm = ZhipuAILLM(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4",
    temperature=0
)
response = index.query(query, llm=llm)
print(response)


# 方法二：RetrievalQA + retriever
# 分步/处理大型文档
# 数据加载
docs = loader.load()
# 分块，当前文档小省略分块

# embedding
embeddings = HuggingFaceEmbeddings(model_name=model_name)
# embed = embeddings.embed_query("Hi my name is Harrison")
# print(len(embed))
# print(embed[:5])

# 向量存储
db = DocArrayInMemorySearch.from_documents(
    docs,
    embedding=embeddings,
)

query = "Please suggest a shirt with sunblocking"
docs = db.similarity_search(query)
# print(len(docs))
# print(docs[0])

# 回答问题
# 创建检索器通用接口
retriever = db.as_retriever()
# 不需要自定义一个LLM类
llm = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0
)

# qdocs = "".join([docs[i].page_content for i in range(len(docs))])  # 将合并文档中的所有页面内容到一个变量中

# 通过LangChain链封装
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # stuff:将所有内容组合成一个文档
    retriever=retriever,
    verbose=True,
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.invoke({"query": query})
# print(type(response))   #dict
print(response['result'])