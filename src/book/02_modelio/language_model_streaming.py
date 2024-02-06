# ← モジュールをインポート
from langchain_openai import ChatOpenAI
# ユーザーからのメッセージであるHumanMessageをインポート
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# 環境変数
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ChatOpenAIを初期化する
chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
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
