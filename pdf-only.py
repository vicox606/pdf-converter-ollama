from langchain_ollama.llms import OllamaLLM
from langchain_core.prompt import ChatPromptTemplate

model = OllamaLLM(model="llama3.2")

template = """
You Are Exerpt in reviewing  documents

here are an revelant question : {reviews}

here is the question to answer : {question}

"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model
result = chain.invoke({ "reviews": [], "question": "what is the main topic of the document?" })
print(result) 