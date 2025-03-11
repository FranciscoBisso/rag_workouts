"""HANDLES PRIVATE DATA INDEXING"""

# pylint: disable=C0411 # disable wrong-import-order rule from pylint
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
from pathlib import Path
from pydantic import SecretStr
from rich.progress import track
from transformers import AutoTokenizer  # type: ignore # <- mypy issue #1198
from typing import List, TypedDict

# .VENV VARIABLES
load_dotenv()

# DIRECTORIES
current_dir: Path = Path().cwd()
bibliography: Path = current_dir / "bibliography"
vector_store_persistent_dir: str = str(
    current_dir / "database" / "actuacion_del_abogado_en_la_causa_judicial"
)
storage_dir: Path = (
    current_dir / "parent_documents" / "actuacion_del_abogado_en_la_causa_judicial"
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

# INITIALIZE TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(hf_model)

# SPLITTER TO CREATE CHILD DOCS
child_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "\n", "!", "?", ".", ";", ":", ",", " "],
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

# VECTOR STORE TO INDEX THE CHILD DOCS
child_vector_store: Chroma = Chroma(
    collection_name="child_docs",
    embedding_function=hf_embeddings,
    persist_directory=vector_store_persistent_dir,
)

# STORAGE LAYER FOR PARENT DOCUMENTS
file_store: LocalFileStore = LocalFileStore(root_path=storage_dir)
docstore = create_kv_docstore(file_store)

# INITIALIZE THE RETRIEVER
retriever: ParentDocumentRetriever = ParentDocumentRetriever(
    child_splitter=child_splitter,
    vectorstore=child_vector_store,
    docstore=docstore,
)


class FileMetadata(TypedDict):
    """FILE'S METADATA"""

    name: str
    path: Path


def files_finder(dir_path: Path | str, file_type: str = "md") -> List[FileMetadata]:
    """
    SEARCHES AND RETRIEVES FILES' METADATA FROM A SPECIFIED DIRECTORY
        ARGS:
            dir_path (Path | str): TARGET DIRECTORY PATH TO SEARCH FILES
            file_type (str): FILE EXTENSION TO FILTER (DEFAULT: 'md')

        RETURNS:
            List[FileMetadata]: LIST OF DICTIONARIES CONTAINING FILE INFO:
                - NAME: FILENAME
                - PATH: COMPLETE FILE PATH

        RAISES:
            ValueError:
                - IF DIRECTORY DOESN'T EXIST
                - IF DIRECTORY IS EMPTY
                - IF NO FILES WITH SPECIFIED EXTENSION ARE FOUND
    """

    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise ValueError(f"❌ files_finder() => DIRECTORY ({dir_path}) DOESN'T EXIST.")

    if not any(dir_path.iterdir()):
        raise ValueError(f"❌ files_finder() => DIRECTORY ({dir_path}) IS EMPTY.")

    if not file_type.startswith("."):
        file_type = f".{file_type}"

    # SEARCH FOR REQUIRED FILES
    files_info: List[FileMetadata] = [
        FileMetadata(name=f.name, path=f)
        for f in track(
            dir_path.glob(f"*{file_type}"),
            description="[bold cyan]🔍 SEARCHING FILES[/]",
            total=len(list(dir_path.glob(f"*{file_type}"))),
        )
        if f.is_file()
    ]

    # CHECK IF FILES WERE FOUND
    if not files_info:
        raise ValueError(
            f"❌ files_finder() => NO FILES OF TYPE ({file_type}) WERE FOUND IN DIRECTORY ({dir_path})."
        )

    return files_info


def directory_loader(
    directory_path: str | Path, file_type: str = "md"
) -> List[Document]:
    """
    LOADS AND PROCESSES DOCUMENTS FROM A SPECIFIED DIRECTORY
        ARGS:
            directory_path (str | Path): PATH TO THE DOCUMENTS' DIRECTORY
            file_type (str, optional): FILE EXTENSION TO FILTER. DEFAULTS TO "md"

        RETURNS:
            List[Document]: LIST OF PROCESSED DOCUMENTS SORTED BY TITLE

        RAISES:
            ValueError: IF DIRECTORY DOESN'T EXIST OR NO FILES ARE FOUND
    """

    files: List[FileMetadata] = files_finder(directory_path, file_type=file_type)

    loaded_docs: List[Document] = []
    for file in track(
        files,
        description="📂 [bold yellow]LOADING FILES[/]",
        total=len(files),
    ):
        loaded_docs.append(
            Document(
                page_content=file["path"].read_text(encoding="utf-8"),
                metadata={
                    "source": str(file["path"]),
                    "title": file["path"].stem,
                },
            )
        )

    loaded_docs.sort(key=lambda doc: doc.metadata["title"])

    return loaded_docs


def split_by_headers(loaded_docs: List[Document]) -> List[Document]:
    """SPLITS BASED ON MARKDOWN HEADINGS WITH PROGRESS TRACKING."""

    if not loaded_docs:
        raise ValueError("❌ split_by_headers() >>> NO DOCS TO SPLIT.")

    headers = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
        ("#####", "H5"),
        ("######", "H6"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    docs_splitted_by_headers: List[Document] = []

    for loaded_doc in track(
        loaded_docs,
        total=len(loaded_docs),
        description="[bold orange1]🪚 SPLITTING DOCS[/]",
    ):
        chunks_splitted_by_headers: List[Document] = splitter.split_text(
            loaded_doc.page_content
        )

        for chunk in chunks_splitted_by_headers:
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
    """
    INDEXES DOCUMENTS IN BATCHES INTO A PARENT DOCUMENT RETRIEVER
        ARGS:
            documents (List[Document]): LIST OF DOCUMENTS TO BE INDEXED
            parent_docs_retriever (ParentDocumentRetriever): RETRIEVER TO INDEX THE DOCUMENTS INTO
            batch_size (int, OPTIONAL): NUMBER OF DOCUMENTS TO PROCESS PER BATCH. DEFAULTS TO 10

        RETURNS:
            ParentDocumentRetriever: THE RETRIEVER WITH THE INDEXED DOCUMENTS

        RAISES:
            ValueError: IF NO DOCUMENTS ARE PROVIDED OR IF NO RETRIEVER IS PROVIDED
    """

    if not documents:
        raise ValueError("❌ feed_retriever() >>> MISSING DOCS TO INDEX.")

    if not documents:
        raise ValueError("❌ feed_retriever() >>> MISSING RETRIEVER.")

    total_documents = len(documents)
    for item in track(
        range(0, total_documents, batch_size),
        description="🗄️ [bold violet]INDEXING BATCHES[/]",
    ):
        batch = documents[item : min(item + batch_size, total_documents)]
        parent_docs_retriever.add_documents(batch, ids=None)

    return retriever


if __name__ == "__main__":
    loaded_files: List[Document] = directory_loader(bibliography)
    parent_docs = split_by_headers(loaded_files)
    # parent_retriever = feed_retriever(parent_docs, retriever)
