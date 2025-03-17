"""HANDLER FOR THE PDF FILES"""

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
from pydantic import SecretStr
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


def load_files(uploaded_files: List[UploadedFile]) -> List[List[Document]]:
    """
    PROCESSES UPLOADED PDF FILES INTO DOCUMENT OBJECTS FOR RAG OPERATIONS
        ARGS:
            uploaded_files: LIST OF PDF FILES UPLOADED BY THE USER

        RETURNS:
            List[List[Document]]: NESTED LIST WHERE EACH INNER LIST CONTAINS THE PAGES
            OF A SINGLE PDF AS DOCUMENT OBJECTS

        RAISES:
            ValueError: IF NO PDF FILES ARE PROVIDED
    """

    if not uploaded_files:
        raise ValueError("load_files() >>> MISSING PDF FILES")

    parsed_files: List[List[Document]] = []

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for pdf_file in track(
            uploaded_files, description="[bold sea_green1]LOADING PDF FILES[/]"
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
            parsed_files.append(file_pages_as_documents)

    return parsed_files


async def feed_retriever(
    loaded_files: List[List[Document]],
    parent_docs_retriever: ParentDocumentRetriever = retriever,
    batch_size: int = 10,
) -> ParentDocumentRetriever:
    """SETUPS THE RETRIEVER & INDEXES DOCUMENTS INTO THE VECTOR STORE.
    IF loaded_files IS TOO LARGE, parent_docs_retriever.add_documents() WONT BE ABLE TO HANDLE IT.
    THUS, IT NEEDS TO BE SPLIT INTO BATCHES.
        ARGS:
            - loaded_files: NESTED LIST OF DOCUMENT OBJECTS FROM PDF FILES
            - parent_docs_retriever: RETRIEVER INSTANCE FOR DOCUMENT INDEXING
            - batch_size: NUMBER OF DOCUMENTS TO PROCESS IN EACH BATCH

        RETURNS:
            ParentDocumentRetriever: CONFIGURED RETRIEVER WITH INDEXED DOCUMENTS

        RAISES:
            ValueError: IF NO DOCUMENTS ARE PROVIDED FOR INDEXING
    """
    if not loaded_files:
        raise ValueError("feed_retriever() >>> MISSING DOCS TO INDEX.")

    print("[bold cyan]INDEXING:[/]")
    for loaded_file in loaded_files:
        file_pages = len(loaded_file)
        for item in track(
            range(0, file_pages, batch_size),
            description=f"[bold cyan]  - {loaded_file[0].metadata['title']}[/]",
        ):
            batch = loaded_file[item : min(item + batch_size, file_pages)]
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
    docs_from_uploaded_files: List[List[Document]] = load_files(uploaded_bibliography)
    parent_retriever: ParentDocumentRetriever = await feed_retriever(
        loaded_files=docs_from_uploaded_files, parent_docs_retriever=retriever
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
