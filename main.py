import os
import sys

from typing import Dict, List, TypedDict, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import START, END, StateGraph

from langchain_community.document_loaders import TextLoader

from dotenv import load_dotenv

load_dotenv()

checkpointer = MemorySaver()

loader = TextLoader("./sample.txt", encoding="utf-8")
documents = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_documents(documents)

store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class RAGState(TypedDict):
    query: str
    context: Optional[List[Document]]
    answer: Optional[str]

def retrieve(state: RAGState) -> RAGState:
    docs = store.similarity_search(state["query"])
    return {"context": docs}

def generate(state: RAGState) -> RAGState:
    context_text = "\n\n".join(d.page_content for d in state["context"])
    prompt = f"""Answer using only the context below.

            Context:
            {context_text}

            Question: {state["query"]}
            Answer:"""
    response = llm.invoke(prompt)

    return {"answer": response.content}

builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

result = graph.invoke({"query": "What is Seattle known for?"}, config=config)
print(f"\nQ: {result['query']}")
print(f"A: {result['answer']}")