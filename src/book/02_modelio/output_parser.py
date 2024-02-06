from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, field_validator


import os
from dotenv import load_dotenv

# 環境変数
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
)


class Smartphone(BaseModel):  # ← Pydanticのモデルを定義する
    release_date: str = Field(description="スマートフォンの発売日")  # ← Fieldを使って説明を追加する
    screen_inches: float = Field(description="スマートフォンの画面サイズ(インチ)")
    os_installed: str = Field(description="スマートフォンにインストールされているOS")
    sp_model_name: str = Field(description="スマートフォンのモデル名")

    @field_validator("screen_inches")  # ← validatorを使って値を検証する
    # ← validatorの引数には、検証するフィールドと値が渡される
    def validate_screen_inches(cls, field):
        if field <= 0:  # ← screen_inchesが0以下の場合はエラーを返す
            raise ValueError("Screen inches must be a positive number")
        return field


# ← PydanticOutputParserをSmartPhoneモデルで初期化する
parser = PydanticOutputParser(pydantic_object=Smartphone)

result = chat([  # ← Chat modelsにHumanMessageを渡して、文章を生成する
    HumanMessage(content="Androidでリリースしたスマートフォンを1個挙げて"),
    HumanMessage(content=parser.get_format_instructions())
])


# ← PydanticOutputParserを使って、文章をパースする
parsed_result = parser.parse(result.content)

print(f"モデル名: {parsed_result.sp_model_name}")
print(parsed_result)
