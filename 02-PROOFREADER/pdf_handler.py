"""HANDLER FOR THE PDF FILES"""

import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr
from rich import print
from rich.progress import track
from streamlit.runtime.uploaded_file_manager import UploadedFile
from tempfile import mkdtemp, TemporaryDirectory
from transformers import AutoTokenizer  # type: ignore # <- Mypy(import-untyped)
from typing import List

load_dotenv()

current_dir: Path = Path.cwd()
hf_model: str = "intfloat/multilingual-e5-large"
hf_key: str | None = os.getenv("HF_API_KEY")
if not hf_key:
    raise ValueError("HF_API_KEY ENV VARIABLE IS NOT SET")
HF_API_KEY: SecretStr = SecretStr(hf_key)

# INITIALIZE TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(hf_model)

# SPLITTER TO CREATE PARENT DOCS
parent_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", "!", "?", "."],
    is_separator_regex=False,
    strip_whitespace=True,
    length_function=lambda text: len(
        tokenizer.encode(
            text,
            truncation=False,
            add_special_tokens=False,
        )
    ),
)

# SPLITTER TO CREATE CHILD DOCS
child_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "\n", "!", "?", "."],
    is_separator_regex=False,
    strip_whitespace=True,
    length_function=lambda text: len(
        tokenizer.encode(
            text,
            truncation=False,
            add_special_tokens=False,
            max_length=512,
        )
    ),
)

# EMBEDDINGS' MODEL
hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name=hf_model,
)

# TEMPORARY VECTOR STORE TO INDEX THE CHILD DOCS
vector_store_temp_dir = mkdtemp()
child_vector_store: Chroma = Chroma(
    collection_name="child_docs",
    embedding_function=hf_embeddings,
    persist_directory=vector_store_temp_dir,
)

# TEMPORARY STORAGE LAYER FOR PARENT DOCUMENTS
file_store_temp_dir = mkdtemp()
file_store: LocalFileStore = LocalFileStore(root_path=file_store_temp_dir)
docstore = create_kv_docstore(file_store)

# INITIALIZE THE RETRIEVER
retriever: ParentDocumentRetriever = ParentDocumentRetriever(
    parent_splitter=parent_splitter,
    child_splitter=child_splitter,
    vectorstore=child_vector_store,
    docstore=docstore,
)


# PYDANTIC MODELS
class ParsedFile(BaseModel):
    """
    REPRESENTS A SINGLE FILE WITH ITS PAGES AS DOCUMENT OBJECTS
    """

    pages: List[Document] = Field(description="List of pages as Document objects")


class Library(BaseModel):
    """
    REPRESENTS A COLLECTION OF ParsedFile
    """

    parsed_files_collection: List[ParsedFile] = Field(
        description="List of parsed files in the library"
    )


def load_files(uploaded_files: List[UploadedFile]) -> Library:
    """
    PROCESSES UPLOADED PDF FILES INTO A LIBRARY OF BOOKS
        ARGS:
            uploaded_files: LIST OF PDF FILES UPLOADED BY THE USER

        RETURNS:
            Library: COLLECTION OF ParsedFile WHERE EACH ParsedFile CONTAINS ITS PAGES AS DOCUMENT OBJECTS

        RAISES:
            ValueError: IF NO PDF FILES ARE PROVIDED
    """

    if not uploaded_files:
        raise ValueError("load_files() >>> MISSING PDF FILES")

    files: List[ParsedFile] = []

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for pdf_file in track(
            uploaded_files, description="[bold sea_green2]LOADING PDF FILES[/]"
        ):
            file_path = temp_path / pdf_file.name
            file_path.write_bytes(pdf_file.getvalue())

            file_pages_as_documents: List[Document] = PyMuPDFLoader(file_path).load()

            for page in file_pages_as_documents:
                page.metadata = {
                    "title": file_path.name,
                    "total_pages": page.metadata["total_pages"],
                    "page": page.metadata["page"],
                }

            files.append(ParsedFile(pages=file_pages_as_documents))

    return Library(parsed_files_collection=files)


async def feed_retriever(
    library: Library,
    parent_docs_retriever: ParentDocumentRetriever = retriever,
    batch_size: int = 10,
) -> ParentDocumentRetriever:
    """
    SETUPS THE RETRIEVER & INDEXES DOCUMENTS INTO THE VECTOR STORE.
        ARGS:
            - library: Library instance containing ParsedFile and their pages as Documents
            - parent_docs_retriever: RETRIEVER INSTANCE FOR DOCUMENT INDEXING
            - batch_size: NUMBER OF DOCUMENTS TO PROCESS IN EACH BATCH

        RETURNS:
            ParentDocumentRetriever: CONFIGURED RETRIEVER WITH INDEXED DOCUMENTS

        RAISES:
            ValueError: IF NO DOCUMENTS ARE PROVIDED FOR INDEXING
    """
    if not library.parsed_files_collection:
        raise ValueError("feed_retriever() >>> MISSING DOCS TO INDEX.")

    print("[bold cyan]INDEXING:[/]")
    for parsed_file in library.parsed_files_collection:
        file_pages = len(parsed_file.pages)
        for item in track(
            range(0, file_pages, batch_size),
            description=f"[bold cyan]- {parsed_file.pages[0].metadata['title']}[/]",
        ):
            batch = parsed_file.pages[item : min(item + batch_size, file_pages)]
            parent_docs_retriever.add_documents(batch, ids=None)

    return retriever


async def orchestrate_indexing(
    uploaded_bibliography: List[UploadedFile],
) -> ParentDocumentRetriever:
    """
    COORDINATES THE COMPLETE INDEXING PIPELINE FOR UPLOADED BIBLIOGRAPHY
        ARGS:
            uploaded_bibliography: LIST OF PDF FILES TO BE PROCESSED

        RETURNS:
            ParentDocumentRetriever: FULLY CONFIGURED RETRIEVER WITH INDEXED DOCUMENTS
    """
    library: Library = load_files(uploaded_bibliography)
    parent_retriever: ParentDocumentRetriever = await feed_retriever(
        library=library, parent_docs_retriever=retriever
    )
    return parent_retriever


def index_bibliography(
    uploaded_bibliography: List[UploadedFile],
) -> ParentDocumentRetriever:
    """
    MAIN ENTRY POINT FOR PDF PROCESSING AND INDEXING OPERATIONS
        ARGS:
            uploaded_bibliography: LIST OF PDF FILES TO BE PROCESSED

        RETURNS:
            ParentDocumentRetriever: RETRIEVER READY FOR RAG OPERATIONS
    """

    if not uploaded_bibliography:
        raise ValueError("index_bibliography() >>> MISSING PDF FILES")

    return asyncio.run(orchestrate_indexing(uploaded_bibliography))


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

    if uploaded_bibliography:
        index_bibliography(uploaded_bibliography)
