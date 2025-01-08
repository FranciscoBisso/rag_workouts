"""HANDLER FOR THE LLM INTERACTION"""

from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pydantic import BaseModel
from typing import List, Dict

# LOCAL IMPORTS
from pdf_handler import load_files

load_dotenv()
groq_model: str = "llama-3.1-8b-instant"
llm: ChatGroq = ChatGroq(model=groq_model, temperature=0, stop_sequences=[])


def extract_qa_pairs(exams: List[Document]) -> List[List[Dict]]:
    """EXTRACTS Q&A FROM THE EXAMS"""

    # PYDANTIC MODELS FOR THE RESPONSE
    class QAPair(BaseModel):
        """MODEL FOR A SINGLE Q&A PAIR"""

        consigna: str
        respuesta: str

    class QAPairs(BaseModel):
        """MODEL FOR MULTIPLE Q&A PAIRS"""

        qa_pairs: List[QAPair]

    # INITIALIZE THE OUTPUT PARSER
    parser = JsonOutputParser(pydantic_object=QAPairs)

    exams_qa_pairs: List[List[Dict]] = []
    for exam in exams:
        # TEMPLATE WITH THE EXPECTED FORMAT
        template = (
            "Tu tarea es analizar un examen y extraer todas las consignas junto con sus respectivas respuestas."
            "\n\nInstrucciones específicas:"
            "\n1. Identifica todas las consignas del examen, sin importar si están formuladas como preguntas directas o como instrucciones"
            "\n2. Para cada consigna, extrae la respuesta correspondiente"
            "\n3. Ignora el contenido introductorio como bibliografía o instrucciones generales"
            "\n4. Mantén el texto exacto tanto de las consignas como de las respuestas"
            "\n5. Si hay numeración en las consignas, mantenla"
            "\n\nExamen a analizar:"
            "\n{user_input}"
            "\n\nDevuelve los resultados en el siguiente formato JSON:"
            "\n{format_instructions}"
        )

        # BUILD THE FULL PROMPT
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(
            user_input=exam.page_content,
            format_instructions=parser.get_format_instructions(),
        )

        # INVOKE THE LLM AND PARSE THE RESPONSE
        response = llm.invoke(messages)
        res_content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        parsed_response = parser.parse(res_content)

        exams_qa_pairs.append(parsed_response["qa_pairs"])

    return exams_qa_pairs


def get_answer(exercise: str, retriever: ParentDocumentRetriever) -> str:
    """GETS THE LLM'S RESPONSE"""
    retrieved_docs: List[Document] = retriever.invoke(exercise)
    context: str = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # BUILD THE PROMPT
    prompt_template = ChatPromptTemplate(
        [
            (
                "system",
                "Sos profesor universitario de derecho procesal civil y comercial argentino que responde preguntas a sus alumnos universitarios. "
                + "Tus respuestas deben ser lo más completas, detalladas y exhaustivas posible. "
                + "Integrá y relacioná la información de todo el CONTEXTO proporcionado para dar una respuesta comprehensiva. "
                + "Incluí ejemplos, plazos, efectos y consecuencias cuando estén disponibles en el CONTEXTO proporcionado. "
                + "Estructura la respuesta en párrafos ordenados lógicamente.",
            ),
            (
                "human",
                "Responder la siguiente pregunta ÚNICAMENTE en base al CONTEXTO proporcionado. "
                + "Si la respuesta no se encuentra en el CONTEXTO proporcionado, simplemente respondé: "
                + "'Lo lamento. No tengo información sobre la cuestión planteada'."
                + "\n\nCONTEXTO:\n{context}"
                + "\n\nPREGUNTA: {user_input}",
            ),
        ]
    )

    # BUILD THE CHAIN & GET THE MODEL'S RESPONSE
    chain = prompt_template | llm
    chain_response = chain.invoke({"user_input": exercise, "context": context})
    res_content = (
        chain_response.content
        if isinstance(chain_response.content, str)
        else str(chain_response.content)
    )

    return res_content


def answer_exercises(
    exams_qa_pairs: List[List[Dict]], retriever: ParentDocumentRetriever
) -> List[List[Dict]]:
    """ANSWERS THE EXAMS' QUESTIONS"""

    for exam in exams_qa_pairs:
        for qa_pair in exam:
            qa_pair["ai_answer"] = get_answer(qa_pair["consigna"], retriever)

    return exams_qa_pairs


def get_llm_response(
    uploaded_exams: List[UploadedFile], retriever: ParentDocumentRetriever
) -> List[List[Dict]]:
    """GETS THE LLM'S RESPONSE"""
    exams: List[Document] = load_files(uploaded_exams)
    exams_qa_pairs: List[List[Dict]] = extract_qa_pairs(exams)
    llm_answers: List[List[Dict]] = answer_exercises(exams_qa_pairs, retriever)

    return llm_answers
