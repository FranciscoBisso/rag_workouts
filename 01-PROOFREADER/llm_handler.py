"""HANDLER FOR THE LLM INTERACTION"""

import dotenv
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
llm = ChatGroq(model=groq_model, temperature=0, stop_sequences=[])


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


def extract_qa_pairs(exams: List[Document]) -> List[List[Dict]]:
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
        parsed_response = parser.parse(response.content)

        exams_qa_pairs.append([parsed_response["qa_pairs"]])

    return exams_qa_pairs


def get_llm_response(uploaded_exams: List[UploadedFile]):
    """GETS THE LLM'S RESPONSE"""
    exams: List[Document] = pdf_to_doc(uploaded_exams)
    # TODO: CHECK IF THIS IS THE BEST DATA STRUCTURE FOR THE RESPONSE
    qa_pairs: List[List[Dict]] = extract_qa_pairs(exams)

    return qa_pairs
