import os
from openai import OpenAI
from dotenv import load_dotenv

# 環境変数
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAIのクライアントを作成
client = OpenAI(api_key=OPENAI_API_KEY)  # ←APIキーを指定して、OpenAIのクライアントを作成

# OpenAIのAPIを呼び出して、言語モデルを呼び出す
stream = client.chat.completions.create(  # ←OpenAIのAPIを呼び出すことで、言語モデルを呼び出している
    model="gpt-3.5-turbo",  # ←呼び出す言語モデルの名前
    messages=[
        {
            "role": "user",
            "content": "iPhone8のリリース日を教えて"  # ←入力する文章(プロンプト)
        },
    ],
    stream=True
)

print(stream)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")