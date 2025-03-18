"""
UI MODULE FOR THE PROOFREADER APPLICATION.
THIS MODULE PROVIDES THE STREAMLIT UI COMPONENTS AND LOGIC FOR THE APPLICATION.
"""

import streamlit as st
from typing import List, Optional, Tuple
from streamlit.runtime.uploaded_file_manager import UploadedFile

# LOCAL IMPORTS
from llm_handler import ExamEAATriadsCollection, get_llm_response
from pdf_handler import index_bibliography


def display_comparison(triads_collections: List[ExamEAATriadsCollection]) -> None:
    """
    DISPLAYS A COMPARISON OF STUDENT ANSWERS AND AI ANSWERS
        ARGS:
            triads_collections (List[ExamEAATriadsCollection]): LIST OF COLLECTIONS OF EXERCISE-ANSWER-AI ANSWER TRIADS
    """
    if not triads_collections:
        st.warning("No data to display.")
        return

    # ITERATE THROUGH EACH EXAM'S COLLECTION
    for i, exam_collection in enumerate(triads_collections):
        st.subheader(f"Exam {i + 1}")

        # ITERATE THROUGH EACH TRIAD IN THE COLLECTION
        for j, triad in enumerate(exam_collection.collection):
            with st.expander(f"Exercise {j + 1}", expanded=False):
                st.markdown("### Exercise")
                st.markdown(triad.exercise)

                # CREATE TWO COLUMNS FOR STUDENT AND AI ANSWERS
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Student Answer")
                    st.markdown(triad.student_answer)

                with col2:
                    st.markdown("### AI Answer")
                    st.markdown(triad.ai_answer)


def setup_sidebar() -> Tuple[
    Optional[List[UploadedFile]], Optional[List[UploadedFile]], bool
]:
    """
    SETS UP THE SIDEBAR UI COMPONENTS
        RETURNS:
            Tuple[Optional[List[UploadedFile]], Optional[List[UploadedFile]], bool]: TUPLE CONTAINING UPLOADED EXAMS, UPLOADED BIBLIOGRAPHY FILES, AND PROCESS BUTTON STATE
    """
    with st.sidebar:
        st.subheader("Upload Bibliography")
        uploaded_bibliography = st.file_uploader(
            "Upload bibliography files (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="bibliography_files",
        )

        st.subheader("Upload Exams")
        uploaded_exams = st.file_uploader(
            "Upload exam files (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            key="exam_files",
        )

        process_button = st.button("Process Exams", type="primary")

        return uploaded_exams, uploaded_bibliography, process_button


def main() -> None:
    """
    MAIN FUNCTION THAT SETS UP THE UI AND HANDLES USER INTERACTIONS
    """
    # SET PAGE CONFIG
    st.set_page_config(
        page_title="Proofreader",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # SETUP SIDEBAR AND GET USER INPUTS
    uploaded_exams, uploaded_bibliography, process_button = setup_sidebar()

    # MAIN CONTENT AREA
    st.markdown("# 📝 Proofreader\n##### :gray[_Correcciones rápidas y precisas_]")

    # INITIALIZE SESSION STATE IF NOT ALREADY DONE
    if "processed_results" not in st.session_state:
        st.session_state.processed_results = None

    # PROCESS BUTTON LOGIC
    if process_button:
        if not uploaded_exams:
            st.error("Please upload at least one exam file.")
            return

        if not uploaded_bibliography:
            st.error("Please upload at least one bibliography file.")
            return

        # SHOW PROCESSING MESSAGE
        with st.spinner("Processing exams..."):
            try:
                # INDEX BIBLIOGRAPHY AND CREATE RETRIEVER
                retriever = index_bibliography(uploaded_bibliography)

                # PROCESS EXAMS AND GET RESULTS
                results = get_llm_response(uploaded_exams, retriever)

                # STORE RESULTS IN SESSION STATE
                st.session_state.processed_results = results

                # DISPLAY SUCCESS MESSAGE
                st.success("Exams processed successfully!")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

    # DISPLAY RESULTS IF AVAILABLE
    if st.session_state.processed_results:
        display_comparison(st.session_state.processed_results)
    else:
        st.info(
            "Upload exams and bibliography files, then click 'Process Exams' to start."
        )


if __name__ == "__main__":
    main()
