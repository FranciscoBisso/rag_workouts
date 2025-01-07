"""
Module for handling queries and responses using Langchain.
"""

from dotenv import load_dotenv
from data_indexer import retriever
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List, Iterator

load_dotenv()

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
                "system",
                "Sos profesor universitario de derecho procesal civil y comercial argentino que responde preguntas a sus alumnos. "
                + "Es FUNDAMENTAL que tu respuesta refleje TODA la información disponible en el CONTEXTO PROPORCIONADO, sin omitir ningún detalle."
                + "\n\nPara cada tema mencionado, debes: "
                + "\n1. Explicar exhaustivamente todos sus elementos y características."
                + "\n2. Mencionar y desarrollar todas las variantes y excepciones."
                + "\n3. Citar los artículos específicos de los códigos procesales, leyes y normativa cuando estén disponibles."
                + "\n4. Incluir ejemplos concretos y casos jurisprudenciales si se mencionan."
                + "\n5. Explicar los plazos, procedimientos y consecuencias legales relevantes."
                + "\n6. Establecer conexiones entre los conceptos que se relacionan entre sí."
                + "\n7. Desarrollar las diferencias entre jurisdicciones cuando corresponda."
                + "\n8. Al responder no hagas menciones como 'Según el texto...', 'Conforme a los documentos suministrados...' o expresiones similares."
                + "\n\nTus respuestas deben ser extensas, minuciosas y explicativas, aprovechando cada fragmento de información disponible en el CONTEXTO PROPORCIONADO. "
                + "Estructura la respuesta en párrafos ordenados lógicamente y expande sobre cada punto relevante.",
            ),
            (
                "human",
                "Responder al siguiente mensaje ÚNICAMENTE en base al CONTEXTO PROPORCIONADO. "
                + "Si la respuesta no se encuentra en el CONTEXTO PROPORCIONADO, simplemente responder con: "
                + "'Lo lamento, no poseo conocimiento suficiente para brindarte una respuesta adecuada'."
                + "\n\nPREGUNTA: {user_input}"
                + "\n\nCONTEXTO:\n{context}",
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
    print(f"[Q]: {user_input}".upper())

    print(">> RETRIEVED DOCS:\n\n")
    for i, doc in enumerate(retrieved_docs):
        print(
            f"DOC N°:{i+1}\n\n{doc.metadata["headers"]}\n{doc.page_content}\n\n{"==="*20}",
            end="\n\n",
        )

    if full_message:
        print(f"\n\n>> [A's METADATA]: {full_message}")
