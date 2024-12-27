"""UI for the Proofreader app"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List

# LOCAL IMPORTS
from pdf_handler import handle_pdf
from llm_handler import pdf_to_doc


# PAGE'S CONFIG
st.set_page_config(page_title="Proofreader", page_icon="📑", layout="centered")

# PAGE'S TITLE
st.markdown(
    """# Proofreader
##### :gray[_Correcciones rápidas y precisas_]"""
)


with st.sidebar:
    # BIBLIOGRAPHY UPLOADER WIDGET
    uploaded_bibliography: List[UploadedFile] | None = st.file_uploader(
        label="UPLOAD BIBLIOGRAPHY (.pdf)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_bibliography:
        bibliography: Chroma = handle_pdf(uploaded_bibliography)

    # EXAMS UPLOADER WIDGET
    uploaded_exams: list[UploadedFile] | None = st.file_uploader(
        label="UPLOAD EXAMS (.pdf)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_exams:
        exams: List[Document] = pdf_to_doc(uploaded_exams)
        for i, exam in enumerate(exams):
            print(f"EXAM N°:{i}\n{exam}\n\n{'==='*15}", end="\n\n")
