"""UI for the Proofreader app"""

from langchain_chroma import Chroma
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List, Dict

# LOCAL IMPORTS
from pdf_handler import handle_pdf
from llm_handler import get_llm_response


# PAGE'S CONFIG
st.set_page_config(page_title="Proofreader", page_icon="📑", layout="centered")

# PAGE'S TITLE
st.markdown(
    """# Proofreader
##### :gray[_Correcciones rápidas y precisas_]"""
)

# SIDEBAR
with st.sidebar:
    # BIBLIOGRAPHY UPLOADER WIDGET
    uploaded_bibliography: List[UploadedFile] | None = st.file_uploader(
        label="UPLOAD BIBLIOGRAPHY",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # EXAMS UPLOADER WIDGET
    uploaded_exams: list[UploadedFile] | None = st.file_uploader(
        label="UPLOAD EXAMS",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_bibliography and uploaded_exams:
        res: List[List[Dict]] = get_llm_response(
            uploaded_exams, handle_pdf(uploaded_bibliography)
        )
        for exam in res:
            print("===" * 15, end="\n\n")
            for item in exam:
                print(
                    f"[Q]: {item["consigna"]}\n\n[AL]: {item["respuesta"]}\n\n[AI]: {item["ai_answer"]}{"--" * 30}",
                    end="\n\n",
                )
