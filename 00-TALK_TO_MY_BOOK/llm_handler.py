"""
Module for handling queries and responses using Langchain.
"""

import os
from dotenv import load_dotenv
from data_indexer import retriever
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List, Iterator

load_dotenv()

current_dir: str = os.getcwd()
groq_model: str = "llama-3.1-8b-instant"


def get_response_from_llm(user_input: str) -> Iterator[BaseMessage]:
    """
    BASED ON THE USER'S QUERY:
      - RETRIEVES RELATED DOCUMENTS;
      - GENERATES A PROMPT; AND
      - CALLS THE LLM TO GET A RESPONSE.
    """

    # INITIALIZE THE RETRIEVER AND RETRIEVE THE RELATED DOCS
    retrieved_docs: List[Document] = retriever.invoke(user_input)

    context: str = "\n\n".join(
        [f"{doc.metadata['headers']}\n{doc.page_content}" for doc in retrieved_docs]
    )

    # BUILD THE PROMPT
    prompt_template = ChatPromptTemplate(
        [
            (
                # "system",
                # "Sos profesor universitario de derecho procesal civil y comercial argentino que responde preguntas a sus alumnos. "
                # + "Tus respuestas deben ser lo más completas, detalladas y exhaustivas posible, abarcando todos los aspectos relevantes encontrados en la totalidad del CONTEXTO proporcionado. "
                # + "Integra y relaciona la información de todo el CONTEXTO proporcionado para dar una respuesta completa y comprehensiva. "
                # + "Desarrolla exhaustivamente cada concepto mencionado, incluyendo todos sus aspectos y matices. "
                # + "Proporciona el marco procesal completo cuando sea relevante. "
                # + "Establece conexiones entre los diferentes aspectos del tema tratado. "
                # + "Asegúrate de incluir ejemplos, plazos, efectos y consecuencias cuando estén disponibles en el CONTEXTO proporcionado. "
                # + "Estructura la respuesta en párrafos ordenados lógicamente y expande sobre cada punto relevante.",
                "system",
                "Sos profesor universitario de derecho procesal civil y comercial argentino que responde preguntas a sus alumnos. "
                + "Es FUNDAMENTAL que tu respuesta refleje TODA la información disponible en el CONTEXTO proporcionado, sin omitir ningún detalle. "
                + "Para cada tema mencionado, debes: "
                + "1. Explicar exhaustivamente todos sus elementos y características "
                + "2. Mencionar y desarrollar todas las variantes y excepciones "
                + "3. Citar los artículos específicos de los códigos procesales cuando estén disponibles "
                + "4. Incluir ejemplos concretos y casos jurisprudenciales si se mencionan "
                + "5. Explicar los plazos, procedimientos y efectos legales relevantes "
                + "6. Establecer conexiones con otros conceptos relacionados del derecho procesal "
                + "7. Desarrollar las diferencias entre jurisdicciones cuando corresponda "
                + "Tus respuestas deben ser extensas y minuciosas, aprovechando cada fragmento de información disponible en el contexto. "
                + "Estructura la respuesta en párrafos ordenados lógicamente y expande sobre cada punto relevante.",
            ),
            (
                "human",
                "Responder la siguiente pregunta ÚNICAMENTE en base al CONTEXTO proporcionado. "
                + "Al responder no hagas menciones como 'Según el texto...', 'Conforme a los documentos suministrados...' o expresiones similares. "
                + "Si la respuesta no se encuentra en el CONTEXTO proporcionado, simplemente respondé: "
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
