"""
Module for handling queries and responses using Langchain.
"""

import os
import dotenv
from index_data import persistent_dir, embeddings_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List, Iterator

dotenv.load_dotenv()

current_dir: str = os.getcwd()
vector_store_path: str = persistent_dir
groq_model: str = "llama-3.1-8b-instant"


def get_response_from_llm(user_input: str) -> Iterator[BaseMessage]:
    """
    BASED ON THE USER'S QUERY:
      - RETRIEVES RELATED DOCUMENTS;
      - GENERATES A PROMPT; AND
      - CALLS THE LLM TO GET A RESPONSE.
    """

    # INITIALIZE THE VECTOR STORE AND RETRIEVE THE RELATED DOCS
    db = Chroma(
        persist_directory=vector_store_path, embedding_function=embeddings_model
    )

    retrieved_docs: List[Document] = db.similarity_search(user_input, k=10)

    context: str = "\n\n".join(
        [f"{doc.metadata['headers']}\n{doc.page_content}" for doc in retrieved_docs]
    )

    # BUILD THE PROMPT
    prompt_template = ChatPromptTemplate(
        [
            (
                "system",
                "Sos profesor universitario de derecho procesal civil y comercial argentino que responde preguntas a sus alumnos universitarios. "
                + "Tus respuestas deben ser lo más completas, detalladas y exhaustivas posible, abarcando todos los aspectos relevantes encontrados en el contexto. "
                + "Integrá y relacioná la información de todo el contexto proporcionado para dar una respuesta comprehensiva. "
                + "Incluí ejemplos, plazos, efectos y consecuencias cuando estén disponibles en el contexto proporcionado. "
                + "Estructura la respuesta en párrafos ordenados lógicamente.",
            ),
            (
                "human",
                "Responder la siguiente pregunta ÚNICAMENTE en base al contexto proporcionado. "
                + "Al responder no hagas menciones como 'Según el texto...', 'Conforme a los documentos suministrados...' o expresiones similares. "
                + "Si la respuesta no se encuentra en el contexto proporcionado, simplemente respondé: "
                + "'Lo lamento. No tengo información sobre la cuestión planteada'."
                + "\n\nCONTEXTO:\n{context}"
                + "\n\nPREGUNTA: {user_input}",
            ),
        ]
    )

    # DEFINE THE LLM MODEL
    llm_model = ChatGroq(
        model=groq_model, temperature=0, verbose=True, streaming=True, stop_sequences=[]
    )

    # BUILD THE CHAIN & GET THE MODEL'S RESPONSE
    chain = prompt_template | llm_model
    chain_streamed_response = chain.stream(
        {"user_input": user_input, "context": context}
    )

    full_message = None
    for chunk in chain_streamed_response:
        full_message = chunk
        yield chunk

    # PRINT USEFUL INFO
    print(f"[Q]: {user_input}")
    for i, doc in enumerate(retrieved_docs):
        print(
            f"DOC N°:{i+1}\n{doc.metadata["headers"]}\n{doc.page_content}\n\n{"==="*20}\n"
        )

    if full_message:
        print(f"\n[A's METADATA]: {full_message}")
