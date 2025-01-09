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

        exercise: str
        student_answer: str

    class QAPairs(BaseModel):
        """MODEL FOR MULTIPLE Q&A PAIRS"""

        qa_pairs: List[QAPair]

    # INITIALIZE THE OUTPUT PARSER
    parser = JsonOutputParser(pydantic_object=QAPairs)

    exams_qa_pairs: List[List[Dict]] = []
    for exam in exams:
        template = (
            "Your task is to analyze an exam and extract all exercises along with their respective answers."
            "\n\nSpecific instructions:"
            "\n1. Identify all exercises in the exam, regardless of whether they are formulated as direct questions or as instructions"
            "\n2. For each exercise, extract the corresponding answer. "
            "If there is no corresponding answer, fill the answer field with the text 'Sin respuesta'."
            "\n3. Ignore introductory content such as bibliography or general instructions"
            "\n4. Keep the exact text of both the exercises and answers"
            "\n5. If the exercises are numbered, maintain such numbering. "
            "For example:\n1) First exercise prompt.\n2) Second exercise prompt.\n3) ..."
            "\n\nExam to analyze:"
            "\n{user_input}"
            "\n\nReturn the results in the following JSON format:"
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
                "Eres un asistente virtual experto en derecho procesal civil y comercial argentino."
                + " Es FUNDAMENTAL que tu respuesta refleje TODA la información disponible en el CONTEXTO PROPORCIONADO, sin omitir ningún detalle."
                + "\n\nPara cada tema mencionado, debes:"
                + "\n1) Explicar exhaustivamente todos sus elementos y características."
                + "\n2) Mencionar y desarrollar todas las variantes y excepciones."
                + "\n3) Citar los artículos específicos de los códigos procesales, leyes y normativa cuando estén disponibles."
                + "\n4) Incluir ejemplos concretos y casos jurisprudenciales si se mencionan."
                + "\n5) Explicar los plazos, procedimientos y consecuencias legales relevantes."
                + "\n6) Establecer conexiones entre los conceptos que se relacionan entre sí."
                + "\n7) Desarrollar las diferencias entre jurisdicciones cuando corresponda."
                + "\n8) Al responder NO hagas menciones como 'Según el texto proporcionado...', 'Conforme a los documentos suministrados...' o expresiones similares."
                + " Responde como si la información del CONTEXTO PROPORCIONADO fuera tuya y no de un libro."
                + "\n\nTus respuestas deben ser extensas, minuciosas y explicativas, aprovechando cada fragmento de información disponible en el CONTEXTO PROPORCIONADO."
                + "Estructura la respuesta en párrafos ordenados lógicamente y expande sobre cada punto relevante."
                + " No repitas en un párrafo lo que ya has dicho en otro.",
            ),
            (
                "human",
                "Responder el siguiente ejercicio ÚNICAMENTE en base al CONTEXTO PROPORCIONADO."
                + " Si la respuesta no se encuentra en el CONTEXTO PROPORCIONADO, simplemente responder con:"
                + " 'Lo lamento, no poseo conocimiento suficiente para brindarte una respuesta adecuada'."
                + "\n\nEJERCICIO: {user_input}"
                + "\n\nCONTEXTO:\n{context}",
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
            qa_pair["ai_answer"] = get_answer(qa_pair["exercise"], retriever)

    return exams_qa_pairs


def get_llm_response(
    uploaded_exams: List[UploadedFile], retriever: ParentDocumentRetriever
) -> List[List[Dict]]:
    """GETS THE LLM'S RESPONSE"""
    exams: List[Document] = load_files(uploaded_exams)
    exams_qa_pairs: List[List[Dict]] = extract_qa_pairs(exams)
    llm_answers: List[List[Dict]] = answer_exercises(exams_qa_pairs, retriever)

    return llm_answers
