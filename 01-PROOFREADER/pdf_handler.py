"""HANDLER FOR THE PDF FILES"""

# from io import BytesIO
import os
import tempfile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import List


current_dir: str = os.getcwd()
hf_model: str = "intfloat/multilingual-e5-large"


def load_files(uploaded_bibliography: List[UploadedFile]) -> List[List[Document]]:
    """
    ARGS: uploaded_bibliography (List[UploadedFile])
    RETURNS: List[List[Document]]. EACH INNER LIST CORRESPONDS TO A PDF FILE
    """
    print("LOADING PDF FILES...")

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

            for i, doc in enumerate(documents):
                doc.metadata["source"] = (
                    pdf_file.name.strip()
                    .upper()
                    .replace(".PDF", ".pdf")
                    .replace(". ", "_")
                )
                # if i < 10:
                #     print(
                #         f"\tDOC N° {i+1}\nLEN(CONTENT): {len(doc.page_content)}\n\n{'==='*15}\n"
                #     )

            all_documents.append(documents)

    return all_documents


def split_by_tokens(bibliography: List[List[Document]]) -> List[Document]:
    """
    ARGS: bibliography (List[List[Document]])
    RETURNS: List[Document]. ALL THE DOCUMENTS SPLIT BY TOKENS INTO SMALLER CHUNKS
    """

    print("SPLITTING BIBLIOGRAPHY BY TOKENS...")

    if not bibliography:
        raise ValueError("Ups! No se encontró bibliografía a ser dividida")

    documents: List[Document] = []
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=hf_model, tokens_per_chunk=500, chunk_overlap=12
    )

    for book in bibliography:
        docs = splitter.split_documents(book)
        documents.extend(docs)

    for i, doc in enumerate(documents):
        if i < 10:
            print(
                f"\tDOC N° {i+1}\nLEN(CONTENT): {len(doc.page_content)}\n\n{'==='*15}\n"
            )

    return documents


def count_tokens(documents: List[Document]):
    """
    ARGS: documents (List[Document])
    RETURNS: int. MIN & MAX NUMBER OF TOKENS IN THE DOCUMENTS
    """
    print("COUNTING TOKENS...")

    counter = SentenceTransformersTokenTextSplitter(model_name=hf_model)

    tokens_per_doc: List[int] = [
        counter.count_tokens(text=doc.page_content) for doc in documents
    ]

    print(f"MIN: {min(tokens_per_doc)}\nMAX: {max(tokens_per_doc)}")
