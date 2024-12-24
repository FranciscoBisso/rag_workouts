"""HANDLER FOR THE PDF FILES"""

# from io import BytesIO
import os
import tempfile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List


current_dir: str = os.getcwd()


def load_files(uploaded_bibliography: List[UploadedFile]) -> List[List[Document]]:
    """
    ARGS: uploaded_bibliography (List[UploadedFile])

    RETURNS: List[List[Document]]. EACH INNER LIST CORRESPONDS TO A PDF FILE"""
    if not uploaded_bibliography:
        raise ValueError("No se encontraron archivos PDF subidos")

    all_documents = []

    # MAKE TEMPORARY DIRECTORY TO SAVE THE PDF FILES
    with tempfile.TemporaryDirectory() as temp_dir:
        for pdf_file in uploaded_bibliography:
            # PDF FILE'S TEMPORARY PATH
            temp_path = os.path.join(temp_dir, pdf_file.name)

            # SAVE TEMPORARY FILE
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getvalue())

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = pdf_file.name

            all_documents.append(documents)

    return all_documents


# if __name__ == "__main__":
# pdf_loader()
