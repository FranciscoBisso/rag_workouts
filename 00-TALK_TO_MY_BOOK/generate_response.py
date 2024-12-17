"""
Module for handling queries and responses using Langchain.
"""

import os
import dotenv
from index_data import persistent_dir, embeddings_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List

dotenv.load_dotenv()

current_dir: str = os.getcwd()
vector_store_path: str = persistent_dir
groq_model: str = "llama-3.1-8b-instant"

def get_response_from_llm(user_input: str) -> str:
    """
    Based on the query provided:
        - retrieves related documents;
        - generates a prompt to be fed to the LLM; and
        - calls the LLM to get a response.
    """

    # llm_model = ChatGroq(model=groq_model, temperature=0, verbose=True)
    llm_model = ChatGroq(model=groq_model, temperature=0, verbose=True, streaming=True)
    
    db = Chroma(
        persist_directory=vector_store_path, embedding_function=embeddings_model
    )

    retrieved_docs: List[Document] = db.similarity_search(user_input, k=10)
    for doc in retrieved_docs:
        print(f"{doc.metadata["headers"]}\n{doc.page_content}\n\n{"==="*20}\n")
    
    context: str = "\n\n".join(
        [f"{doc.metadata['headers']}\n{doc.page_content}" for doc in retrieved_docs]
    )

    prompt_template = ChatPromptTemplate(
        [
            (
                "system",
                "Sos un asistente legal que responde preguntas sobre derecho procesal.",
            ),
            (
                "human",
                "Responder la siguiente pregunta ÚNICAMENTE en base al contexto proporcionado. "
                + "La respuesta debe ser clara, completa y precisa. "
                + "Sin embargo, al responder no hagas menciones como 'Según el texto...' o expresiones similares. "
                + "Respondé en primera persona. "
                + "Si la respuesta no se encuentra en los documentos suministrados, responde con 'Lo lamento. No tengo información sobre la cuestión planteada'."
                + "\n\nCONTEXTO:\n{context}"
                + "\n\nPREGUNTA: {user_input}",
            ),
        ]
    )

    chain = prompt_template | llm_model
    chain_result = chain.stream({"user_input": user_input, "context": context})
    
    return chain_result

