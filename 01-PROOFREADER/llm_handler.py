from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List

# LOCAL IMPORTS
from pdf_handler import load_files


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
