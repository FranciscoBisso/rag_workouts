"""
MAIN MODULE FOR HANDLING LLM (LARGE LANGUAGE MODEL) INTERACTIONS AND PROCESSING EXAM CONTENT.
THIS MODULE PROVIDES FUNCTIONALITY FOR EXTRACTING Q&A PAIRS FROM EXAMS AND GENERATING AI RESPONSES.
"""

# pylint: disable=C0411 # disable wrong-import-order rule from pylint
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from rich.progress import track
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List, Dict

# from rich import print

# LOCAL IMPORTS
from pdf_handler import load_files

load_dotenv()
groq_model: str = "llama-3.1-8b-instant"
llm: ChatGroq = ChatGroq(model=groq_model, temperature=0, stop_sequences=[])


# PYDANTIC MODELS
class QAPair(BaseModel):
    """
    REPRESENTS A SINGLE QUESTION-ANSWER PAIR FROM AN EXAM
    """

    exercise: str = Field(description="The exercise or question text")
    student_answer: str = Field(description="The student's answer to the exercise")


class QAPairs(BaseModel):
    """
    REPRESENTS A COLLECTION OF QUESTION-ANSWER PAIRS
    """

    qa_pairs: List[QAPair]


def build_prompt(system_msg: str, human_msg: str) -> ChatPromptTemplate:
    """
    CREATES A ChatPromptTemplate WITH SYSTEM AND HUMAN MESSAGES
        ARGS:
            system_msg (str): THE SYSTEM INSTRUCTION MESSAGE
            human_msg (str): THE HUMAN INPUT MESSAGE

        RETURNS:
            ChatPromptTemplate: A FORMATTED CHAT PROMPT TEMPLATE
    """
    return ChatPromptTemplate([("system", system_msg), ("human", human_msg)])


def extract_qa_pairs(exams: List[List[Document]]) -> List[List[Dict]]:
    """
    PROCESSES EXAM DOCUMENTS TO EXTRACT QUESTION-ANSWER PAIRS
        ARGS:
            exams (List[List[Document]]): LIST OF EXAM DOCUMENTS TO PROCESS

        RETURNS:
            List[List[Dict]]: NESTED LIST OF DICTIONARIES CONTAINING EXTRACTED QA PAIRS
    """

    # INITIALIZE THE OUTPUT PARSER
    parser = JsonOutputParser(pydantic_object=QAPairs)

    exams_qa_pairs: List[List[Dict]] = []

    for exam in track(
        exams,
        description="[bold light_coral]EXTRACTING QA PAIRS FROM EXAMS[/]",
        total=len(exams),
    ):
        exam_content: str = "\n\n".join([page.page_content for page in exam])

        system_msg: str = (
            "1. TO DO:"
            "\nYour task is to analyze an exam and extract all exercises along with their respective answers."
            "\n\n2. FORMAT INSTRUCTIONS:"
            "\nReturn the results in the following JSON format:"
            "\n{format_instructions}"
            "\n\n3. SPECIFIC TASK INSTRUCTIONS:"
            "\n\t3.1. Identify all exercises in the exam, regardless of whether they are formulated as direct questions or as instructions"
            "\n\t3.2. For each exercise, extract the corresponding answer."
            " If there is no corresponding answer, fill the answer field with the text 'Sin respuesta'."
            "\n\t3.3. Ignore introductory content such as bibliography or general instructions"
            "\n\t3.4. Keep the exact text of both the exercises and answers"
            "\n\t3.5. If the exercises are numbered, maintain such numbering. For example:"
            "\n\t\t1) First exercise prompt.\n\t\t2) Second exercise prompt.\n\t\t3) ..."
        )
        human_msg: str = "\n\nPlease analyze the following exam:\n{exam_content}"

        # BUILD THE FULL PROMPT
        prompt = build_prompt(system_msg=system_msg, human_msg=human_msg)
        messages = prompt.format_messages(
            exam_content=exam_content,
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
    """
    GENERATES AN AI RESPONSE FOR A GIVEN EXERCISE USING THE RETRIEVER
        ARGS:
            exercise (str): THE EXERCISE TEXT TO GENERATE AN ANSWER FOR
            retriever (ParentDocumentRetriever): THE RETRIEVER INSTANCE FOR CONTEXT LOOKUP

        RETURNS:
            str: THE AI-GENERATED ANSWER FOR THE EXERCISE
    """

    retrieved_docs: List[Document] = retriever.invoke(exercise)
    context: str = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # BUILD THE PROMPT
    system_msg: str = (
        "1. ROL:"
        " Eres un asistente virtual experto en derecho procesal civil y comercial argentino."
        "\n\n2. TAREA:"
        "Responder ÚNICAMENTE en base al CONTEXTO PROPORCIONADO."
        "\n\n3. FORMA DE RESPONDER:"
        "\nEs FUNDAMENTAL que tu respuesta refleje TODA la información disponible en el CONTEXTO PROPORCIONADO, sin omitir ningún detalle."
        " Tus respuestas deben ser extensas, minuciosas y explicativas, aprovechando cada fragmento de información disponible en el CONTEXTO PROPORCIONADO."
        "Estructura la respuesta en párrafos ordenados lógicamente y expande sobre cada punto relevante."
        " No repitas en un párrafo lo que ya has dicho en otro."
        "\n\nPara cada tema mencionado en el CONTEXTO PROPORCIONADO, debes:"
        "\n\t1) Explicar exhaustivamente todos sus elementos y características."
        "\n\t2) Mencionar y desarrollar todas las variantes y excepciones."
        "\n\t3) Citar los artículos específicos de los códigos procesales, leyes y normativa cuando estén disponibles."
        "\n\t4) Incluir ejemplos concretos y casos jurisprudenciales si se mencionan."
        "\n\t5) Explicar los plazos, procedimientos y consecuencias legales relevantes."
        "\n\t6) Establecer conexiones entre los conceptos que se relacionan entre sí."
        "\n\t7) Desarrollar las diferencias entre jurisdicciones cuando corresponda."
        "\n\t8) Al responder NO hagas menciones como 'Según el texto proporcionado...', 'Conforme a los documentos suministrados...' o expresiones similares."
        " Responde como si la información del CONTEXTO PROPORCIONADO fuera tuya y no del CONTEXTO PROPORCIONADO."
        "\n\n4. ADVERTENCIA:"
        "\nSi la respuesta no se encuentra en el CONTEXTO PROPORCIONADO, simplemente responder con:"
        " 'Lo lamento, no poseo conocimiento suficiente para brindarte una respuesta adecuada'."
    )
    human_msg: str = (
        "¿Podrías dar respuesta al siguiente ejercicio acudiendo ÚNICAMENTE al CONTEXTO que te proporciono a continuación?"
        "\n\nEJERCICIO: {exercise}"
        "\n\nCONTEXTO:\n{context}"
    )
    prompt = build_prompt(system_msg=system_msg, human_msg=human_msg)

    # BUILD THE CHAIN & GET THE MODEL'S RESPONSE
    chain = prompt | llm
    chain_response = chain.invoke({"exercise": exercise, "context": context})
    res_content = (
        chain_response.content
        if isinstance(chain_response.content, str)
        else str(chain_response.content)
    )

    return res_content


def answer_exercises(
    exams_qa_pairs: List[List[Dict]], retriever: ParentDocumentRetriever
) -> List[List[Dict]]:
    """
    PROCESSES ALL EXERCISES IN THE QA PAIRS TO GENERATE AI ANSWERS
        ARGS:
            exams_qa_pairs (List[List[Dict]]): NESTED LIST OF QA PAIRS TO PROCESS
            retriever (ParentDocumentRetriever): THE RETRIEVER INSTANCE FOR CONTEXT LOOKUP

        RETURNS:
            List[List[Dict]]: UPDATED QA PAIRS WITH AI-GENERATED ANSWERS
    """

    for exam in track(
        exams_qa_pairs,
        description="[bold sky_blue2]LLM ANSWERING EXERCISES[/]",
        total=len(exams_qa_pairs),
    ):
        for qa_pair in exam:
            qa_pair["ai_answer"] = get_answer(qa_pair["exercise"], retriever)

    return exams_qa_pairs


def get_llm_response(
    uploaded_exams: List[UploadedFile], retriever: ParentDocumentRetriever
) -> List[List[Dict]]:
    """
    MAIN FUNCTION TO PROCESS UPLOADED EXAMS AND GENERATE LLM RESPONSES
        ARGS:
            uploaded_exams (List[UploadedFile]): LIST OF UPLOADED EXAM FILES
            retriever (ParentDocumentRetriever): THE RETRIEVER INSTANCE FOR CONTEXT LOOKUP

        RETURNS:
            List[List[Dict]]: COMPLETE PROCESSED RESULTS WITH EXERCISES, STUDENT ANSWERS, AND AI RESPONSES
    """

    exams: List[List[Document]] = load_files(uploaded_exams)
    exams_qa_pairs: List[List[Dict]] = extract_qa_pairs(exams)
    llm_answers: List[List[Dict]] = answer_exercises(exams_qa_pairs, retriever)

    return llm_answers
