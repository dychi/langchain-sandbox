import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings  # ← OpenAIEmbeddingsをインポート
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter


# 環境変数
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)


chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

prompt = PromptTemplate(template="""文章を元に質問に答えてください。 

文章: 
{document}

質問: {query}
""", input_variables=["document", "query"])


text_splitter = SpacyTextSplitter(chunk_size=300, pipeline="ja_core_news_sm")


@cl.on_chat_start
async def on_chat_start():
    # ファイルが選択されているか確認する変数
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            max_size_mb=20,
            content="PDFを選択してください",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()
    file = files[0]
    print(file.name)
    # if not os.path.exists("tmp"):
    #     os.mkdir("tmp")
    # with open(f"tmp/{file.name}", "wb") as f:  # ← PDFファイルを保存する
    #     f.write(file)  # ← ファイルの内容を書き込む

    documents = PyMuPDFLoader(file.path).load()  # ← 保存したPDFファイルを読み込む
    splitted_documents = text_splitter.split_documents(documents)

    database = Chroma(
        embedding_function=embeddings,
        # 今回はpersist_directoryを指定しないことでデータベースの永続化を行わない
    )

    database.add_documents(splitted_documents)

    cl.user_session.set(  # ← データベースをセッションに保存する
        "database",  # ← セッションに保存する名前
        database  # ← セッションに保存する値
    )
    await cl.Message(content=f"`{file.name}`の読み込みが完了しました。質問を入力してください。").send()


@cl.on_message
async def on_message(input_message: cl.Message):
    print("入力されたメッセージ： " + input_message.content)
    input_message_str = input_message.content
    database = cl.user_session.get("database")  # ← セッションからデータベースを取得する
    documents = database.similarity_search(input_message_str)

    documents_string = ""

    for document in documents:
        documents_string += f"""
    ---------------------------
    {document.page_content}
    """

    result = chat([
        HumanMessage(
            content=prompt.format(
                document=documents_string,
                query=input_message_str
            )
        )
    ])
    await cl.Message(content=result.content).send()
