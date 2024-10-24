"""
Module for handling queries and responses using Langchain and Chroma.
"""

import os
from langchain_chroma import Chroma
from index_data import persistent_dir, embeddings_model


current_dir: str = os.getcwd()
vector_store_path: str = persistent_dir
# question: str = "¿Qué elementos componen el ciclo procedimental?"  # <-- temporary
# question: str = (
#     "¿Qué diferencia hay entre la instancia y la actuación simple?"  # <-- temporary
# )
question: str = "¿Cuáles son los pasos del iter recursivo?"  # <-- temporary


def retrieve_related_docs(query: str):
    """Retrieves relevant documents based on the query."""

    db = Chroma(
        persist_directory=vector_store_path, embedding_function=embeddings_model
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
        # search_type="similarity_score_threshold",
        # search_kwargs={"k": 8, "score_threshold": 0.7},  # <-- tops 0.7
        # search_type="mmr",
        # search_kwargs={"k": 8, "fetch_k": 24, "lambda_mult": 0.7},
    )

    related_docs = retriever.invoke(query)

    return related_docs


if __name__ == "__main__":
    relevant_docs = retrieve_related_docs(question)
    print(f"\n-> VECTOR STORE: {'/'.join(persistent_dir.split('/')[-2:])}\n")
    for i, doc in enumerate(relevant_docs):
        print(
            f">>> DOC {i+1}:\n\n{doc.metadata['headers']}\n\n{doc.page_content}\n\n{'==='*20}\n"
        )
