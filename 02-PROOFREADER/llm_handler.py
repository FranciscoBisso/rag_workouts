"""
MAIN MODULE FOR HANDLING LLM (LARGE LANGUAGE MODEL) INTERACTIONS AND PROCESSING EXAM CONTENT.
THIS MODULE PROVIDES FUNCTIONALITY FOR EXTRACTING Q&A PAIRS FROM EXAMS AND GENERATING AI RESPONSES.
"""

# pylint: disable=C0411 # disable wrong-import-order rule from pylint
import asyncio
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from rich.progress import track
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List


# LOCAL IMPORTS
from pdf_handler import load_files

load_dotenv()
groq_model: str = "llama-3.1-8b-instant"
llm: ChatGroq = ChatGroq(model=groq_model, temperature=0, stop_sequences=[])


# PYDANTIC MODELS
class EAPair(BaseModel):
    """
    REPRESENTS A SINGLE EXERCISE - STUDENT ANSWER PAIR
    """

    exercise: str = Field(description="Exercise or question text")
    student_answer: str = Field(description="Student's answer to the exercise")


class ExamEAPairsCollection(BaseModel):
    """
    REPRESENTS A COLLECTION OF EXERCISE - STUDENT ANSWER PAIRS
    """

    collection: List[EAPair]


class EAATriad(BaseModel):
    """
    REPRESENTS A SINGLE EXERCISE - STUDENT ANSWER - AI ANSWER TRIAD
    """

    exercise: str = Field(description="Exercise or question text")
    student_answer: str = Field(description="Student's answer to the exercise")
    ai_answer: str = Field(description="LLM's answer to the exercise")


class ExamEAATriadsCollection(BaseModel):
    """
    REPRESENTS A COLLECTION OF EXERCISE - STUDENT ANSWER - AI ANSWER TRIADS
    """

    collection: List[EAATriad]


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


async def get_collections_of_ea_pairs(
    exams: List[List[Document]],
) -> List[ExamEAPairsCollection]:
    """
    ASYNCHRONOUSLY PROCESSES EXAM DOCUMENTS TO EXTRACT QUESTION-ANSWER PAIRS
        ARGS:
            exams (List[List[Document]]): LIST OF EXAM DOCUMENTS TO PROCESS

        RETURNS:
            List[List[Dict]]: LIST OF ExamEAPairsCollection OBJECTS CONTAINING EXTRACTED QA PAIRS
    """

    collections_of_triads: List[ExamEAPairsCollection] = []

    for exam in track(
        exams,
        description="[bold yellow]EXTRACTING TRIADS FROM EXAMS[/]",
        total=len(exams),
    ):
        exam_content: str = "\n\n".join([page.page_content for page in exam])

        system_msg: str = (
            "1. TO DO:"
            "\nYour task is to analyze an exam and extract all exercises along with their respective answers."
            "\n\n2. SPECIFIC TASK INSTRUCTIONS:"
            "\n\t- Identify all exercises in the exam, regardless of whether they are formulated as direct questions or as instructions"
            "\n\t- For each exercise, extract the corresponding answer."
            " If there is no corresponding answer, fill the answer field with the text 'Sin respuesta'."
            "\n\t- Ignore introductory content such as bibliography or general instructions"
            "\n\t- Keep the exact text of both the exercises and answers"
            "\n\t- If the exercises are numbered, maintain such numbering. For example:"
            "\n\t\t1) First exercise prompt.\n\t\t2) Second exercise prompt.\n\t\t3) ..."
        )
        human_msg: str = "\n\nPlease analyze the following exam:\n{exam_content}"

        # BUILD THE FULL PROMPT
        prompt = build_prompt(system_msg=system_msg, human_msg=human_msg)

        # CREATE A STRUCTURED LLM USING with_structured_output
        structured_llm = llm.with_structured_output(ExamEAPairsCollection)

        # BUILD THE CHAIN & INVOKE
        chain = prompt | structured_llm
        response = await chain.ainvoke({"exam_content": exam_content})
        if isinstance(response, ExamEAPairsCollection):
            collections_of_triads.append(response)
        else:
            raise ValueError(
                f"Expected ExamEAPairsCollection, but got {type(response)}"
            )

    return collections_of_triads


async def get_answer(exercise: str, retriever: ParentDocumentRetriever) -> str:
    """
    ASYNCHRONOUSLY GENERATES A CONTEXTUALIZED AI RESPONSE FOR A GIVEN EXERCISE
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
    chain_response = await chain.ainvoke({"exercise": exercise, "context": context})
    res_content = (
        chain_response.content
        if isinstance(chain_response.content, str)
        else str(chain_response.content)
    )

    return res_content


async def answer_exercises(
    exams_ea_pairs_collections: List[ExamEAPairsCollection],
    retriever: ParentDocumentRetriever,
) -> List[ExamEAATriadsCollection]:
    """
    ASYNCHRONOUSLY PROCESSES ALL EXERCISES IN THE QA PAIRS TO GENERATE AI ANSWERS
        ARGS:
            exams_ea_pairs_collections (List[ExamEAPairsCollection]): NESTED LIST OF EXERCISE-ANSWER PAIRS TO PROCESS
            retriever (ParentDocumentRetriever): THE RETRIEVER INSTANCE FOR CONTEXT LOOKUP

        RETURNS:
            List[ExamEAATriadsCollection]: NESTED LIST OF EXERCISE-ANSWER-AI ANSWER TRIADS
    """
    exams_triads_collections: List[ExamEAATriadsCollection] = []
    for exam in track(
        exams_ea_pairs_collections,
        description="[bold light_coral]LLM ANSWERING EXERCISES[/]",
        total=len(exams_ea_pairs_collections),
    ):
        exam_new_collection = ExamEAATriadsCollection(collection=[])
        exams_triads_collections.append(exam_new_collection)

        # CREATE TASKS FOR ALL EXAM'S QUESTIONS
        tasks = [get_answer(ea_pair.exercise, retriever) for ea_pair in exam.collection]

        # RUN TASKS CONCURRENTLY
        ai_answers = await asyncio.gather(*tasks)

        # BUILD TRIADS
        for i, ea_pair in enumerate(exam.collection):
            triad = EAATriad(
                exercise=ea_pair.exercise,
                student_answer=ea_pair.student_answer,
                ai_answer=ai_answers[i],
            )
            exam_new_collection.collection.append(triad)

    return exams_triads_collections


async def orchestrate_generative_process(
    uploaded_exams: List[UploadedFile], retriever: ParentDocumentRetriever
):
    """
    ASYNCHRONOUSLY PROCESSES UPLOADED EXAMS AND GENERATES AN LLM RESPONSE
        ARGS:
            uploaded_exams (List[UploadedFile]): LIST OF UPLOADED EXAM FILES
            retriever (ParentDocumentRetriever): THE RETRIEVER INSTANCE FOR CONTEXT LOOKUP

        RETURNS:
            List[ExamEAATriadsCollection]: COMPLETE PROCESSED RESULTS AS A LIST OF LIST OF EXERCISE-ANSWER-AI ANSWER TRIADS
    """

    exams: List[List[Document]] = load_files(uploaded_exams)
    exams_collections_of_ea_pairs: List[
        ExamEAPairsCollection
    ] = await get_collections_of_ea_pairs(exams)
    llm_answers = await answer_exercises(exams_collections_of_ea_pairs, retriever)

    return llm_answers


def get_llm_response(
    uploaded_exams: List[UploadedFile], retriever: ParentDocumentRetriever
):
    """
    SYNCHRONOUS WRAPPER THAT EXECUTES THE ASYNCHRONOUS VERSION OF THE TASK ORCHESTRATOR
        ARGS:
            uploaded_exams (List[UploadedFile]): LIST OF UPLOADED EXAM FILES
            retriever (ParentDocumentRetriever): THE RETRIEVER INSTANCE FOR CONTEXT LOOKUP

        RETURNS:
            List[ExamEAATriadsCollection]: COMPLETE PROCESSED RESULTS FROM THE ASYNCHRONOUS TASK ORCHESTRATOR
    """
    return asyncio.run(orchestrate_generative_process(uploaded_exams, retriever))
