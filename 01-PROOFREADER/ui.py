"""UI for the Proofreader app"""

import os
from langchain_core.documents import Document
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List

# LOCAL IMPORTS
from pdf_handler import load_files, split_by_tokens, count_tokens

current_dir: str = os.getcwd()

# PAGE'S CONFIG
st.set_page_config(page_title="Proofreader", page_icon="📑", layout="centered")

# PAGE'S TITLE
st.markdown(
    """# Proofreader
##### :gray[_Correcciones rápidas y precisas_]"""
)


with st.sidebar:
    # BIBLIOGRAPHY UPLOADER
    uploaded_bibliography: List[UploadedFile] | None = st.file_uploader(
        label="BIBLIOGRAPHY (.pdf)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_bibliography:
        bibliography: List[List[Document]] = load_files(uploaded_bibliography)
        documents: List[Document] = split_by_tokens(bibliography)
        minimax_tokens: str = count_tokens(documents)

    # EXAMS UPLOADER
    uploaded_exams: list[UploadedFile] | None = st.file_uploader(
        label="EXAMS (.pdf)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_exams:
        exams = load_files(uploaded_exams)
