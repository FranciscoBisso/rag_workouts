"""
Module for handling the indexing process of the markdown files.
"""

import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter


current_dir: str = os.getcwd()
md_dir: str = os.path.join(current_dir, "private_data")
persistent_dir: str = os.path.join(
    current_dir, "database", "actuacion_del_abogado_en_la_causa_judicial"
)


def directory_loader(directory_path: str) -> List[Document]:
    """Loads markdown documents from a given directory."""
    print("1. LOADING DOCUMENTS FROM DIRECTORY...")

    loaded_docs: List[Document] = []

    for root, _, files in os.walk(directory_path):

        md_files = [f for f in files if f.endswith(".md")]

        for i, filename in enumerate(md_files):
            file_path = os.path.join(root, filename)
            directory_name = os.path.basename(root)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                document = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "directory": directory_name,
                        "filename": filename.replace(".md", ""),
                    },
                )
                loaded_docs.append(document)

                print(f" 1.{i+1}. loading document -> {filename}")

    loaded_docs.sort(key=lambda doc: doc.metadata["filename"])

    return loaded_docs


def split_by_headers(loaded_docs: Document) -> List[Document]:
    """Splits based on markdown headings."""
    print("\n2. SPLITTING DOCUMENTS BY HEADERS...")

    headers = [
        ("#", "Encabezado 1"),
        ("##", "Encabezado 2"),
        ("###", "Encabezado 3"),
        ("####", "Encabezado 4"),
        ("#####", "Encabezado 5"),
        ("######", "Encabezado 6"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=True
    )

    docs_splitted_by_headers: List[Document] = []

    for loaded_doc in loaded_docs:
        chunks_splitted_by_headers: List[Document] = splitter.split_text(
            loaded_doc.page_content
        )

        for chunk in chunks_splitted_by_headers:
            chunk.metadata = {**loaded_doc.metadata, "headers": chunk.metadata}
            docs_splitted_by_headers.append(chunk)

    return docs_splitted_by_headers


def split_by_paragraphs(docs_splitted_by_headers: List[Document]) -> List[Document]:
    """Splits based on paragraphs."""
    print("\n3. SPLITTING DOCUMENTS BY PARAGRAPHS...")

    docs_splitted_by_paragraphs: List[Document] = []

    for doc in docs_splitted_by_headers:
        paragraphs: List[str] = doc.page_content.split("\n")

        for paragraph in paragraphs:
            final_doc = Document(page_content=paragraph, metadata=doc.metadata)
            docs_splitted_by_paragraphs.append(final_doc)

    return docs_splitted_by_paragraphs


def stringify_headers(chunk_headers: dict) -> str:
    """Converts a dictionary to a string."""

    formatted_headers: str = ""

    for value in chunk_headers:
        formatted_headers += f"{chunk_headers[value]}\n"

    return formatted_headers


def move_headers_into_chunks_content(list_of_docs: List[Document]) -> List[Document]:
    """Add headers to the chunks content."""
    print("\n4. ENHANCING CHUNKS...")

    docs_with_headers: List[Document] = []
    for chunk in list_of_docs:
        headers = chunk.metadata["headers"]
        content = f"{stringify_headers(headers)}\n{chunk.page_content}".strip()

        del chunk.metadata["headers"]

        full_doc = Document(metadata=chunk.metadata, page_content=content)
        docs_with_headers.append(full_doc)

    return docs_with_headers


def chunks_lengths(chunks: List[Document]) -> str:
    """Returns the size of each chunk."""

    sizes = [len(chunk.page_content) for chunk in chunks]
    unrepeated_lengths = sorted(set(sizes))
    min_len = min(sizes)
    max_len = max(sizes)

    return f"- MIN: {min_len}\n+ MAX: {max_len}\n-> ALL: {unrepeated_lengths}"


def generate_embeddings(chunks_to_embed: List[Document]) -> Chroma:
    """Embeds document splits into the vector store."""
    print("\n5. EMBEDDING CHUNKS...")

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks_to_embed,
        embedding=embeddings_model,
        persist_directory=persistent_dir,
    )

    return vector_store


if __name__ == "__main__":
    loaded_files = directory_loader(md_dir)

    chunks_splitted_by_md_headers = split_by_headers(loaded_files)

    chunks_splitted_by_paragraphs = split_by_paragraphs(chunks_splitted_by_md_headers)

    chunks_with_headers = move_headers_into_chunks_content(
        chunks_splitted_by_paragraphs
    )

    lengths = chunks_lengths(chunks_with_headers)

    embeddings_generator = generate_embeddings(chunks_with_headers)

    # for index, val in enumerate(chunks_with_headers):
    #     print(f"""DOC N° {index}:\n{val.metadata}\n\n{'==='*20}\n""")

    # print(f"LENGTHS:\n{lengths}")
