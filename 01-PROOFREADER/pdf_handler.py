"""HANDLER FOR THE PDF FILES"""

import os
import tempfile
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pydantic import SecretStr
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List

load_dotenv()

current_dir: str = os.getcwd()
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
    add_start_index=True,
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
    add_start_index=True,
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
vector_store_temp_dir = tempfile.mkdtemp()
child_vector_store: Chroma = Chroma(
    collection_name="child_docs",
    embedding_function=hf_embeddings,
    persist_directory=vector_store_temp_dir,
)

# TEMPORARY STORAGE LAYER FOR PARENT DOCUMENTS
file_store_temp_dir = tempfile.mkdtemp()
file_store: LocalFileStore = LocalFileStore(root_path=file_store_temp_dir)
docstore = create_kv_docstore(file_store)


# INITIALIZE THE RETRIEVER
retriever: ParentDocumentRetriever = ParentDocumentRetriever(
    parent_splitter=child_splitter,
    child_splitter=child_splitter,
    vectorstore=child_vector_store,
    docstore=docstore,
)


def load_files(uploaded_bibliography: List[UploadedFile]) -> List[Document]:
    """
    CONVERTS USER'S UPLOADED PDF FILES TO List[List[Document]] WHERE EACH INNER LIST CORRESPONDS TO A PDF FILE
    """

    if not uploaded_bibliography:
        raise ValueError("load_files() >>> MISSING PDF FILES")

    all_documents = []

    # MAKE TEMPORARY DIRECTORY TO SAVE THE PDF FILES
    with tempfile.TemporaryDirectory() as temp_dir:
        for pdf_file in tqdm(
            uploaded_bibliography, desc="LOADING PDF FILES", unit="file"
        ):
            # PDF FILE'S TEMPORARY PATH
            temp_path = os.path.join(temp_dir, pdf_file.name)

            # SAVE TEMPORARY FILE
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getvalue())

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = (
                    pdf_file.name.strip()
                    .upper()
                    .replace(".PDF", ".pdf")
                    .replace(" ", "_")
                )

            all_documents.extend(documents)

    return all_documents


def feed_retriever(
    documents: List[Document],
    parent_docs_retriever: ParentDocumentRetriever,
    batch_size: int = 10,
) -> ParentDocumentRetriever:
    """SETUPS THE RETRIEVER WITH PROGRESS TRACKING."""

    if not documents:
        raise ValueError("feed_retriever() >>> MISSING DOCS TO INDEX.")

    if not documents:
        raise ValueError("feed_retriever() >>> MISSING RETRIEVER.")

    total_documents = len(documents)
    for item in tqdm(
        range(0, total_documents, batch_size), desc="INDEXING BATCHES", unit="batch"
    ):
        batch = documents[item : min(item + batch_size, total_documents)]
        parent_docs_retriever.add_documents(batch, ids=None)

    return retriever


def index_bibliography(
    uploaded_bibliography: List[UploadedFile],
) -> ParentDocumentRetriever:
    """
    MAIN FUNCTION TO INDEX THE UPLOADED PDF FILES.
    """

    docs: List[Document] = load_files(uploaded_bibliography)
    parent_retriever: ParentDocumentRetriever = feed_retriever(
        documents=docs, parent_docs_retriever=retriever
    )

    return parent_retriever
