import os
import sys
import wikipediaapi

from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, END, StateGraph

from dotenv import load_dotenv

load_dotenv()

wiki = wikipediaapi.Wikipedia(
    user_agent="MetadataEmbeddingTutorial/1.0",
    language="en"
)

BANDS = ["The Beatles", "The Cure"]
raw_docs: List[Document] = []

for title in BANDS:
    page = wiki.page(title)

    if not page.exists():
        print(f"  WARNING: '{title}' not found on Wikipedia, skipping.")
        continue

    doc = Document(
        page_content=page.text,
        metadata={
            "title":page.title,
            "url":page.fullurl,
            "source":"wikipedia"
        }
    )

    raw_docs.append(doc)
    print(f"  ✓ {page.title:20s} — {len(page.text):,} chars")

if not raw_docs:
    sys.exit("ERROR: No documents fetched. Check your internet connection.")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def create_vectorstore(
    documents: List[Document],
    collection_name: str,
    metadata_fields_to_embed: Optional[List[str]] = None,
) -> Chroma:
    """
    Split documents into chunks and index them into a Chroma vector store.
 
    Args:
        documents:               Raw LangChain Documents.
        collection_name:         Chroma collection name (must be unique per run).
        metadata_fields_to_embed:
            When provided, the value of each listed metadata field is
            prepended to the chunk text before embedding.
 
            Example — with metadata_fields_to_embed=["title"]:
                original chunk:  "...visited Bangor in August 1967..."
                embedded text:   "title: The Beatles\n\n...visited Bangor..."
 
            This anchors the embedding to the document's identity even when
            the chunk text itself doesn't mention the band by name.
 
    Returns:
        A ready-to-query Chroma vector store.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[".","\n"," "],
    )
    
    chunks = splitter.split_documents(documents)

    if metadata_fields_to_embed:
        enriched = []
        for chunk in chunks:
            prefix = " | ".join(
                f"{k}: {chunk.metadata.get(k,'')}" for k in metadata_fields_to_embed
            )
            enriched.append(
                Document(
                    page_content=f"{prefix}\n\n{chunk.page_content}",
                    metadata=chunk.metadata,
                )
            )
        chunks = enriched
        print(f" (title prepended to each chunks)", end="")

    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )
    return store

store_standard = create_vectorstore(
    raw_docs,
    collection_name="standard"
)

store_with_metadata = create_vectorstore(
    raw_docs,
    collection_name="with_metadata",
    metadata_fields_to_embed=["title"]
)

class RetrievalState(TypedDict):
    query: str
    docs_standard: Optional[List[Document]]
    docs_with_metadata: Optional[List[Document]]

TOP_K = 3

def retrieve_standard(state: RetrievalState) -> RetrievalState:
    """Node: retrieve from the store WITHOUT embedded metadata."""
    retriever = store_standard.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    return {"docs_standard": retriever.invoke(state["query"])}

def retrieve_with_metadata(state: RetrievalState) -> RetrievalState:
    """Node: retrieve from the store WITH embedded metadata"""
    retriever = store_with_metadata.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    return {"docs_with_metadata": retriever.invoke(state["query"])}



builder = StateGraph(RetrievalState)

builder.add_node("retrieve_standard", retrieve_standard)
builder.add_node("retrieve_with_metadata", retrieve_with_metadata)

builder.set_entry_point("retrieve_standard")

builder.add_edge(START, "retrieve_standard")
builder.add_edge("retrieve_standard", "retrieve_with_metadata")
builder.add_edge("retrieve_with_metadata", END)

retrieval_graph = builder.compile()

SEPARATOR = "=" * 68

def print_doc(index: int, doc: Document) -> None:
    title = doc.metedata.get("title", "unknown")
    snippet = doc.page_content.replace("\n", " ")[:130]
    print(f"\n [{index}] title : {title}")
    print(f"       chunk : {snippet}....")

def run_comparison(query: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  QUERY: {query}")
    print(SEPARATOR)

    result = retrieval_graph.invoke({"query": query})
    print("\n  ── Standard retriever (no metadata embedded) ──")
    for i, doc in enumerate(result["docs_standard"], 1):
        print_doc(i, doc)

    print("\n  ── Retriever WITH title metadata embedded ──")
    for i, doc in enumerate(result["docs_with_metadata"], 1):
        print_doc(i, doc)

run_comparison("Have the Beatles ever been to Bangor?")
run_comparison("What announcements did the band The Cure make in 2022?")
