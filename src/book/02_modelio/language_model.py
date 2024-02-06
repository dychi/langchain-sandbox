import os
from dotenv import load_dotenv
# ← モジュールをインポート
from langchain_openai import ChatOpenAI
# ユーザーからのメッセージであるHumanMessageをインポート
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate

# 環境変数
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか？",
    input_variables=[
        "product"
    ],
)

messages = [
    HumanMessage(content=prompt.format(product="iPhone8")),
    # HumanMessage(content="iPhone8のリリース日を教えて")
]

result = chat(messages)

print(result.content)
