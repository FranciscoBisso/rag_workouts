"""
Module for handling the indexing process of the markdown files.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

current_dir: str = os.getcwd()
md_dir: str = os.path.join(os.getcwd(), "private_data")


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


def split_by_paragraphs(
    docs_splitted_by_headers: List[Document],
) -> List[Document]:
    """Splits based on paragraphs."""

    print("\n3. SPLITTING DOCUMENTS BY PARAGRAPHS...")
    docs_splitted_by_paragraphs: List[Document] = []

    for doc in docs_splitted_by_headers:
        paragraphs: List[str] = doc.page_content.split("\n")

        for paragraph in paragraphs:
            final_doc = Document(
                page_content=paragraph,
                metadata=doc.metadata,
            )
            docs_splitted_by_paragraphs.append(final_doc)

    return docs_splitted_by_paragraphs


def convert_headers(chunk_headers: dict) -> str:
    """Converts a dictionary to a string."""
    
    formatted_headers: str = ""
    
    for value in chunk_headers:
        formatted_headers += f"{chunk_headers[value]}\n"
        
    return formatted_headers

def add_headers_to_chunks_content(
    list_of_docs: List[Document],
) -> List[Document]:
    """Add headers to the chunks content."""

    for chunk in list_of_docs: 
        headers = chunk.metadata["headers"]
        chunk.page_content = f"{convert_headers(headers)}\n{chunk.page_content}"
        
    return list_of_docs
        


if __name__ == "__main__":
    loaded_docs = directory_loader(md_dir)

    chunks_splitted_by_headers = split_by_headers(loaded_docs)

    chunks_splitted_by_paragraphs = split_by_paragraphs(chunks_splitted_by_headers)
    
    chunks_with_headers = add_headers_to_chunks_content(chunks_splitted_by_paragraphs)

    # for i, document in enumerate(splitted_by_headers):
    # for i, document in enumerate(splitted_by_paragraphs):
    #     print(
    #         f"""DOC N° {i}:\n\n- METADATA:\n{document.metadata["headers"]}\n\n- CONTENT:\n{document.page_content[:100]}\n\n{'==='*20}\n"""
    #     )
    for i, document in enumerate(chunks_with_headers):
        print(
            f"""DOC N° {i}:\n{document.page_content}\n\n{'==='*20}\n"""
        )
