from langchain.prompts import PromptTemplate

promptTemplate = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか？",
    input_variables=[
        "product"
    ],
)

print(promptTemplate)
print(promptTemplate.format(product="iPhone8"))
