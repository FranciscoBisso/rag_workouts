"""
Module for handling queries and responses using Langchain and Chroma.
"""

import os
from typing import List
import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from index_data import persistent_dir, embeddings_model

dotenv.load_dotenv()

current_dir: str = os.getcwd()
vector_store_path: str = persistent_dir
groq_model: str = "llama-3.1-70b-versatile"

question: str = "¿Qué elementos componen el ciclo procedimental?"  # <-- temporary
# question: str = (
#     "¿Qué diferencia hay entre la instancia y la actuación simple?"  # <-- temporary
# )
# question: str = "¿Cuáles son los pasos del iter recursivo?"  # <-- temporary


def retrieve_related_docs(query: str) -> List[Document]:
    """Retrieves related documents based on the query."""

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


def get_response(query: str, related_docs: List[Document]) -> str:
    """Generates a response to the query provided."""

    combined_input = (
        "Acá tenes algunos documentos que te pueden ayudar a responder la pregunta: "
        + query
        + "\n\nDocumentos relevantes:\n"
        + "\n\n".join(
            [f"{doc.metadata['headers']}\n{doc.page_content}" for doc in related_docs]
        )
        + "\n\nPor favor, responde solo con la información proporcionada en los documentos relevantes. "
        + "Si la respuesta no se encuentra en los documentos relevantes, responde con 'No tengo información sobre eso'."
    )

    model = ChatGroq(model=groq_model, temperature=0, verbose=True)

    messages = [
        SystemMessage(
            content="Sos un asistente legal que responde preguntas sobre derecho procesal."
        ),
        HumanMessage(content=combined_input),
    ]

    result = model.invoke(messages)

    return result.content


if __name__ == "__main__":
    relevant_docs = retrieve_related_docs(question)
    response = get_response(question, relevant_docs)

    print(f"\n-> VECTOR STORE: {'/'.join(persistent_dir.split('/')[-2:])}\n")

    for i, doc in enumerate(relevant_docs):
        print(
            f">>> DOC {i+1}:\n\n{doc.metadata['headers']}\n\n{doc.page_content}\n\n{'==='*20}\n"
        )

    print(f"\n=> ANSWER: {response}\n")
