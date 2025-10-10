import os
# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from zai import ZhipuAiClient
from langchain_community.chat_models import ChatZhipuAI
from langchain.prompts import ChatPromptTemplate


client = ZhipuAiClient(api_key=os.getenv("ZHIPU_API_KEY"))
customer_style = """American English in a calm and respectful tone"""
customer_email = """Arrr, I be fuming that me blender lid flew off and splattered me kitchen walls with smoothie! And to make matters worse, the warranty don't cover the cost of cleaning up me kitchen. I need yer help right now, matey!"""


# no langchain
def get_completion(prompt, model='glm-4.5'):
    message = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=message,
        temperature=0.5
    )
    return response.choices[0].message.content

# response = get_completion('What is 1+1?')
# print(response)

template_string = f"""Translate the text that is delimited by triple backticks into a style that is {customer_style}. text:```{customer_email}```"""
# response = get_completion(template_string)
# print(response)

# langchain
chat = ChatZhipuAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    model="glm-4.5",
    temperature=0.5
)

template_string = """Translate the text that is delimited by triple backticks into a style that is {style}. text:```{text}```"""
prompt_template = ChatPromptTemplate.from_template(template_string)
# print(prompt_template)

customer_message = prompt_template.format_messages(
    style=customer_style,
    text=customer_email,
)

# customer_response = chat.invoke(customer_message)
# print('----------use langchain--------------')
# print(customer_response.content)

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

# 指定输出格式
review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

prompt_template = ChatPromptTemplate.from_template(review_template)
# print(prompt_template)

message = prompt_template.format_messages(text=customer_review)
# response = chat.invoke(message)
# print(response.content)
# print(type(response.content))

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(review_template_2)

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")

delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")

price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema,
                    delivery_days_schema,
                    price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

message = prompt.format_messages(text=customer_review, format_instructions=format_instructions)
# print(message)

response = chat.invoke(message)
print(response.content)

output_dict = output_parser.parse(response.content)
print(output_dict)
print(type(output_dict))
print(output_dict.get('delivery_days'))