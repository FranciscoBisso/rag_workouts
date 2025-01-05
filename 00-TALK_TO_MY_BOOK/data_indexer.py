"""HANDLES PRIVATE DATA INDEXING"""

import os
from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import SecretStr
from tqdm import tqdm
from typing import List

load_dotenv()
# DIRECTORIES RELATED VARIABLES
current_dir: str = os.getcwd()
md_files_dir: str = os.path.join(current_dir, "private_data")
storage_dir: str = os.path.join(current_dir, "private_data", "parent_documents")
vector_store_persistent_dir: str = os.path.join(
    current_dir, "database", "actuacion_del_abogado"
)

# HUGGING FACE RELATED VARIABLES
hf_model: str = "intfloat/multilingual-e5-large"
hf_key: str | None = os.getenv("HF_API_KEY")
if not hf_key:
    raise ValueError("HF_API_KEY ENV VARIABLE IS NOT SET")
HF_API_KEY: SecretStr = SecretStr(hf_key)
hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name=hf_model,
)


# SPLITTER TO CREATE CHILD DOCS
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?"],
    is_separator_regex=False,
    add_start_index=True,
)

# VECTOR STORE TO INDEX THE CHILD DOCS
child_vector_store = Chroma(
    collection_name="child_docs",
    embedding_function=hf_embeddings,
    persist_directory=vector_store_persistent_dir,
)

# STORAGE LAYER FOR PARENT DOCUMENTS
file_store = LocalFileStore(root_path=storage_dir)
docstore = create_kv_docstore(file_store)

# INITIALIZE THE RETRIEVER
retriever = ParentDocumentRetriever(
    child_splitter=child_splitter,
    vectorstore=child_vector_store,
    docstore=docstore,
)


def directory_loader(directory_path: str) -> List[Document]:
    """LOADS MARKDOWN DOCUMENTS FROM A GIVEN DIRECTORY WITH PROGRESS INDICATOR."""

    if not os.path.exists(directory_path):
        raise ValueError(
            f"directory_loader() >>> DIRECTORY {directory_path} DOESN'T EXIST."
        )

    loaded_docs: List[Document] = []

    # Primero encontramos todos los archivos .md
    md_files_info = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                md_files_info.append((file_path, filename))

    # Procesamos los archivos con barra de progreso
    for file_path, filename in tqdm(md_files_info, desc="LOADING FILES", unit="File"):
        file_source = "/".join(element for element in file_path.split("/")[-4:])

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            document = Document(
                page_content=content,
                metadata={
                    "source": file_source,
                    "filename": filename.replace(".md", ""),
                },
            )
            loaded_docs.append(document)

    loaded_docs.sort(key=lambda doc: doc.metadata["filename"])

    return loaded_docs


def split_by_headers(loaded_docs: List[Document]) -> List[Document]:
    """SPLITS BASED ON MARKDOWN HEADINGS WITH PROGRESS TRACKING."""

    if not loaded_docs:
        raise ValueError("split_by_headers() >>> NO DOCS TO SPLIT.")

    headers = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
        ("#####", "H5"),
        ("######", "H6"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=True
    )

    docs_splitted_by_headers: List[Document] = []

    for loaded_doc in loaded_docs:
        chunks_splitted_by_headers: List[Document] = splitter.split_text(
            loaded_doc.page_content
        )

        for chunk in tqdm(
            chunks_splitted_by_headers,
            desc=f"SPLITTING {loaded_doc.metadata.get('filename')}",
            unit="chunk",
        ):
            chunk.metadata = {**loaded_doc.metadata, "headers": chunk.metadata}

            formatted_headers: str = ""
            for value in chunk.metadata["headers"]:
                formatted_headers += f"{chunk.metadata['headers'][value]} | "

            chunk.metadata["headers"] = formatted_headers[:-3]
            docs_splitted_by_headers.append(chunk)

    return docs_splitted_by_headers


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
        range(0, total_documents, batch_size), desc="INDEXING", unit="batch"
    ):
        batch = documents[item : min(item + batch_size, total_documents)]
        parent_docs_retriever.add_documents(batch, ids=None)

    return retriever


if __name__ == "__main__":
    loaded_files: List[Document] = directory_loader(md_files_dir)
    parent_docs = split_by_headers(loaded_files)
    parent_retriever = feed_retriever(parent_docs, retriever)
