"""UI for the Proofreader app"""

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List, Dict

# LOCAL IMPORTS
from pdf_handler import index_bibliography
from llm_handler import get_llm_response


def display_qa_item(qa_item: Dict, question_num: int):
    """
    DISPLAY A SINGLE QUESTION-ANSWER ITEM
    ARGS:
        qa_item (Dict): DICT CONTAINING QUESTION, STUDENT ANSWER & AI ANSWER
        question_num (int): QUESTION NUMBER
    """
    with st.expander(f"Question {question_num}", expanded=False):
        # Question
        st.markdown("#### Question")
        st.markdown(
            # f"<div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>{qa_item['question']}</div>",
            # unsafe_allow_html=True,
            {qa_item["exercise"]}
        )

        # Student Answer
        st.markdown("#### Student Answer")
        st.markdown(
            # f"<div style='background-color: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>{qa_item['student_answer']}</div>",
            # unsafe_allow_html=True,
            {qa_item["student_answer"]}
        )

        # AI Answer
        st.markdown("#### AI Answer")
        st.markdown(
            # f"<div style='background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>{qa_item['ai_answer']}</div>",
            # unsafe_allow_html=True,
            {qa_item["ai_answer"]}
        )


def display_llm_response(exams_list: List[List[Dict]]):
    """
    DISPLAY THE LLM'S RESPONSE: A LIST OF EXAMS, WHERE EACH EXAM CONTAINS MULTIPLE QA PAIRS
    ARGS:
        exams_list (List[List[Dict]]): LIST OF EXAMS, EACH CONTAINING A LIST OF QA DICTIONARIES
    """
    # Create tabs for each exam
    tabs = st.tabs([f"Exam {i+1}" for i in range(len(exams_list))])

    # Display content for each exam in its respective tab
    for exam_idx, tab in enumerate(tabs):
        with tab:
            st.markdown(f"### Exam {exam_idx + 1}")
            st.markdown("---")

            # Display each question in the exam
            for q_idx, qa_item in enumerate(exams_list[exam_idx], 1):
                display_qa_item(qa_item, q_idx)


# PAGE'S CONFIG
st.set_page_config(page_title="Proofreader", page_icon="📑", layout="centered")

# PAGE'S TITLE
st.markdown(
    """# Proofreader
##### :gray[_Correcciones rápidas y precisas_]"""
)

if "response" not in st.session_state:
    st.session_state["response"] = None

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
            uploaded_exams, index_bibliography(uploaded_bibliography)
        )

        st.session_state["response"] = res


# CENTRAL ELEMENT
if st.session_state["response"]:
    for element in st.session_state["response"]:
        for item_index, item in enumerate(element):
            with st.expander(item["exercise"], expanded=False):
                # Student Answer
                st.markdown("###### Student Answer:")
                st.write(item["student_answer"])

                # AI Answer
                st.markdown("###### AI Answer:")
                st.write(item["ai_answer"])
