import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_community.chat_models import ChatZhipuAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, \
    ConversationSummaryBufferMemory

llm = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0
)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

conversation.predict(input='Hi, my name is liu')
conversation.predict(input='What is 1+1?')
conversation.predict(input='what is my name')
# print(memory.buffer)
# print(memory.load_memory_variables({}))

memory.save_context({'input':'Hi'}, {'output':"What's up"})
# print(memory.load_memory_variables({}))

# 保存最近k词记录
memory = ConversationBufferWindowMemory(k=1)

# 限制token数量
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)

# 保存历史信息摘要
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=400)