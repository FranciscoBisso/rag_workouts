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
HF_API_KEY: SecretStr = SecretStr(os.getenv("HF_API_KEY"))


def load_files(uploaded_bibliography: List[UploadedFile]) -> List[List[Document]]:
    """
    ARGS: uploaded_bibliography (List[UploadedFile])
    RETURNS: List[List[Document]]. EACH INNER LIST CORRESPONDS TO A PDF FILE
    """

    if not uploaded_bibliography:
        raise ValueError("No se encontraron archivos PDF subidos")

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
    ARGS: bibliography (List[List[Document]])
    RETURNS: List[Document]. ALL THE DOCUMENTS SPLIT BY TOKENS INTO SMALLER CHUNKS
    """

    if not bibliography:
        raise ValueError("Ups! No se encontró bibliografía a ser dividida")

    print("SPLITTING BIBLIOGRAPHY BY TOKENS...")

    documents: List[Document] = []
    splitter = SentenceTransformersTokenTextSplitter(
        model_name=hf_model,
        tokens_per_chunk=500,
        chunk_overlap=12,
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


def count_tokens(documents: List[Document]) -> str:
    """
    ARGS: documents (List[Document])
    RETURNS: int. MIN & MAX NUMBER OF TOKENS IN THE DOCUMENTS
    """

    print("COUNTING TOKENS...")

    counter = SentenceTransformersTokenTextSplitter(model_name=hf_model)

    tokens_per_doc: List[int] = [
        counter.count_tokens(text=doc.page_content) for doc in documents
    ]

    return f"\t - MIN: {min(tokens_per_doc)}\n\t - MAX: {max(tokens_per_doc)}"


def index_docs(documents: List[Document]) -> Chroma:
    """
    GENERATES EMBEDDINGS FOR THE DOCUMENTS AND STORES THEM IN A TEMPORARY IN-MEMORY DATABASE.
    """

    if not documents:
        raise ValueError("No se encontraron documentos para indexar.")

    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_API_KEY,
        model_name=hf_model,
    )

    # VERIFY API CONNECTION OF EMBEDDING'S MODEL
    try:
        embeddings_model.embed_query("Test query")
        print("API connection successful")
    except Exception as e:
        print(f"API connection failed: {str(e)}")
        raise

    batch_size: int = 100
    texts: List[str] = [doc.page_content for doc in documents]
    metadatas: List[dict] = [doc.metadata for doc in documents]

    # INITIALIZE CHROMA VECTOR STORE
    chroma_vector_store = Chroma(
        collection_name="bibliography",
        embedding_function=embeddings_model,
    )

    print("INDEXING DOCS IN VECTOR STORE...")
    # ADD BATCHES INTO CHROMA VECTOR STORE
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_metadata = metadatas[i : i + batch_size]
        chroma_vector_store.add_texts(texts=batch_texts, metadatas=batch_metadata)
        print(
            f"\t- Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
        )

    print("DOCS SUCCESSFULLY INDEXED IN VECTOR STORE...")
    return chroma_vector_store
