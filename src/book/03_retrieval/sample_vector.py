from langchain_openai import OpenAIEmbeddings  # ← OpenAIEmbeddingsをインポート
from numpy import dot  # ← ベクトルの類似度を計算するためにdotをインポート
from numpy.linalg import norm  # ← ベクトルの類似度を計算するためにnormをインポート

# 環境変数
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ← OpenAIEmbeddingsを初期化する
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# ← 質問をベクトル化
query_vector = embeddings.embed_query("飛行車の最高速度は？")

# ← ベクトルの一部を表示
print(f"ベクトル化された質問: {query_vector[:5]}")

# ← ドキュメント1のベクトルを取得
document_1_vector = embeddings.embed_query("飛行車の最高速度は時速150キロメートルです。")
document_2_vector = embeddings.embed_query(
    "鶏肉を適切に下味をつけた後、中火で焼きながらたまに裏返し、外側は香ばしく中は柔らかく仕上げる。"
)  # ← ドキュメント2のベクトルを取得

# ← ベクトルの類似度を計算
cos_sim_1 = dot(
    query_vector,
    document_1_vector) / (norm(query_vector) * norm(document_1_vector))

print(f"ドキュメント1と質問の類似度: {cos_sim_1}")
cos_sim_2 = dot(
    query_vector,
    document_2_vector) / (norm(query_vector) * norm(document_2_vector))
print(f"ドキュメント2と質問の類似度: {cos_sim_2}")