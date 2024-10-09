"""
Module for handling queries and responses using Langchain and Chroma.
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from index_data import persistent_dir


current_dir: str = os.getcwd()
vector_store_path: str = persistent_dir
question: str = "¿Qué elementos componen el ciclo procedimental?"  # <-- temporary


def retrieve_related_docs(query: str):
    """Retrieves relevant documents based on the query."""
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    db = Chroma(
        persist_directory=vector_store_path, embedding_function=embeddings_model
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
        # search_type="mmr",
        # search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.5},
        # search_type="similarity_score_threshold",
        # search_kwargs={"k": 5, "score_threshold": 0.1}
    )

    related_docs = retriever.invoke(query)

    return related_docs


if __name__ == "__main__":
    relevant_docs = retrieve_related_docs(question)
    print("RELEVANT DOCUMENTS:\n")
    for i, doc in enumerate(relevant_docs):
        print(f"DOC {i}:\n\n{doc.page_content}\n\n{'==='*20}\n")
