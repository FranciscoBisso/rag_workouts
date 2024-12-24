"""UI for the Proofreader app"""

import os
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from index_pdf import pdf_loader

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
    uploaded_bibliography: list[UploadedFile] | None = st.file_uploader(
        label="UPLOAD BIBLIOGRAPHY (.pdf)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_bibliography:
        bibliography_files_paths: list[str] = []
        # SAVE FILES TO THE SPECIFIED DIRECTORY
        for pdf_file in uploaded_bibliography:
            pdf_path: str = os.path.join(
                current_dir, "pdf", "bibliography", pdf_file.name
            )
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            if not os.path.commonpath(
                [pdf_path, os.path.join(current_dir, "pdf", "bibliography")]
            ) == os.path.join(current_dir, "pdf", "bibliography"):
                st.error("There was an error while saving the file. Please try again.")
            else:
                st.success("File saved successfully.")
                bibliography_files_paths.append(pdf_path)

    # HANDLE THE PDF FILES (BIBLIOGRAPHY)
    pdf_loader(bibliography_files_paths)

    # EXAMS UPLOADER
    uploaded_exams: list[UploadedFile] | None = st.file_uploader(
        label="UPLOAD EXAMS (.pdf)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_exams:
        # SAVE FILES TO THE SPECIFIED DIRECTORY
        for exam in uploaded_exams:
            exam_path: str = os.path.join(current_dir, "pdf", "exams", exam.name)
            with open(exam_path, "wb") as f:
                f.write(exam.getbuffer())
