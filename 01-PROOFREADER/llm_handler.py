"""HANDLER FOR THE LLM INTERACTION"""

import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pydantic import BaseModel
from typing import List, Dict

# LOCAL IMPORTS
from pdf_handler import load_files

dotenv.load_dotenv()
groq_model: str = "llama-3.1-8b-instant"
llm: ChatGroq = ChatGroq(model=groq_model, temperature=0, stop_sequences=[])


def pdf_to_doc(uploaded_exams: List[UploadedFile]) -> List[Document]:
    """
    FORMATS THE EXAMS' FILES TO BE READ BY THE LLM
    """
    exams: List[List[Document]] = load_files(uploaded_exams)

    list_of_exams: List[Document] = []
    for i, exam in enumerate(exams):
        doc: Document = Document(
            page_content=" ".join([page.page_content.strip() for page in exam]),
            metadata={"source": exam[i].metadata["source"]},
        )

        list_of_exams.append(doc)

    return list_of_exams


def extract_qa_pairs_from_exams(exams: List[Document]) -> List[List[Dict]]:
    """EXTRACTS Q&A FROM THE EXAMS"""

    # PYDANTIC MODELS FOR THE RESPONSE
    class QAPair(BaseModel):
        consigna: str
        respuesta: str

    class ExamResponse(BaseModel):
        qa_pairs: List[QAPair]

    # INITIALIZE THE OUTPUT PARSER
    parser = JsonOutputParser(pydantic_object=ExamResponse)

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


def get_answer(exercise: str, vector_store: Chroma) -> str:
    """GETS THE LLM'S RESPONSE"""
    retrieved_docs: List[Document] = vector_store.similarity_search(
        query=exercise, k=10
    )
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
    exams_qa_pairs: List[List[Dict]], vector_store: Chroma
) -> List[List[Dict]]:
    """ANSWERS THE EXAMS' QUESTIONS"""

    for exam in exams_qa_pairs:
        for qa_pair in exam:
            qa_pair["ai_answer"] = get_answer(qa_pair["consigna"], vector_store)

    return exams_qa_pairs


def get_llm_response(
    uploaded_exams: List[UploadedFile], vector_store: Chroma
) -> List[List[Dict]]:
    """GETS THE LLM'S RESPONSE"""
    exams: List[Document] = pdf_to_doc(uploaded_exams)
    exams_qa_pairs: List[List[Dict]] = extract_qa_pairs_from_exams(exams)
    llm_answers: List[List[Dict]] = answer_exercises(exams_qa_pairs, vector_store)

    return llm_answers
