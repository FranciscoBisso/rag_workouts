"""HANDLER FOR THE PDF FILES"""

import os
import dotenv
import tempfile
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pydantic import SecretStr
from typing import List

dotenv.load_dotenv()

current_dir: str = os.getcwd()
hf_model: str = "intfloat/multilingual-e5-large"
hf_key: str | None = os.getenv("HF_API_KEY")
if not hf_key:
    raise ValueError("HF_API_KEY ENV VARIABLE IS NOT SET")
HF_API_KEY: SecretStr = SecretStr(hf_key)


def load_files(uploaded_bibliography: List[UploadedFile]) -> List[List[Document]]:
    """
    FROM USER'S UPLOADED PDF FILES TO List[List[Document]] WHERE EACH INNER LIST CORRESPONDS TO A PDF FILE
    """

    if not uploaded_bibliography:
        raise ValueError("load_files() >>> MISSING PDF FILES")

    print("LOADING PDF FILES...")

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
                doc.metadata["source"] = (
                    pdf_file.name.strip()
                    .upper()
                    .replace(".PDF", ".pdf")
                    .replace(" ", "_")
                )

            all_documents.append(documents)

    return all_documents


def split_by_tokens(bibliography: List[List[Document]]) -> List[Document]:
    """
    SPLITS DOCUMENTS BY TOKENS INTO SMALLER CHUNKS
    """

    if not bibliography:
        raise ValueError("split_by_tokens() >>> MISSING BIBLIOGRAPHY")

    print("SPLITTING BIBLIOGRAPHY...")

    documents: List[Document] = []
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=hf_model,
        tokens_per_chunk=500,
        chunk_overlap=12,
    )

    for book in bibliography:
        docs = splitter.split_documents(book)
        documents.extend(docs)

    return documents


def count_tokens(documents: List[Document]) -> str:
    """
    COUNTS DOCS' CONTENT BY TOKENS
    """

    print("COUNTING TOKENS...")

    counter = SentenceTransformersTokenTextSplitter(model_name=hf_model)

    tokens_per_doc: List[int] = [
        counter.count_tokens(text=doc.page_content) for doc in documents
    ]

    return f"\t - MIN: {min(tokens_per_doc)}\n\t - MAX: {max(tokens_per_doc)}"


def index_docs(documents: List[Document]) -> Chroma:
    """
    GENERATES EMBEDDINGS FOR THE DOCUMENTS AND STORES THEM IN A TEMPORARY DATABASE.
    """

    if not documents:
        raise ValueError("index_docs() >>> MISSING DOCUMENTS")

    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_API_KEY,
        model_name=hf_model,
    )

    # VERIFY API CONNECTION OF EMBEDDING'S MODEL
    try:
        embeddings_model.embed_query("Test query")
    except Exception as e:
        print(f"index_docs() >>> EMBEDDINGS' API CONNECTION FAILED: {str(e)}")
        raise

    batch_size: int = 50
    texts: List[str] = [doc.page_content for doc in documents]
    metadatas: List[dict] = [doc.metadata for doc in documents]

    # INITIALIZE VECTOR STORE
    vector_store = Chroma(embedding_function=embeddings_model)

    # INDEX DOCS IN BATCHES
    print("INDEXING DOCS IN VECTOR STORE...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metadata = metadatas[i : i + batch_size]
        vector_store.add_texts(texts=batch_texts, metadatas=batch_metadata)
        print(
            f"  - Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
        )

    print("DOCS SUCCESSFULLY INDEXED IN VECTOR STORE...")
    return vector_store


def handle_pdf(uploaded_bibliography: List[UploadedFile]) -> Chroma:
    """
    INDEXES USER'S UPLOADED PDF FILES INTO A VECTOR STORE
    """
    bibliography: List[List[Document]] = load_files(uploaded_bibliography)
    documents: List[Document] = split_by_tokens(bibliography)
    # print(f"TOKENS:\n{count_tokens(documents)}")
    vector_store: Chroma = index_docs(documents)

    return vector_store
