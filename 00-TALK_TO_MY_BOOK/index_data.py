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
    print("1) LOADING DOCUMENTS FROM DIRECTORY...")

    docs_list: List[Document] = []

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
                docs_list.append(document)

                print(f" 1.{i+1}. loading document --> {filename}")

    docs_list.sort(key=lambda doc: doc.metadata["filename"])

    return docs_list


def split_by_headers(doc: Document) -> List[Document]:
    """Splits a single document into chunks based on markdown's header tags."""

    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=True
    )

    list_of_docs_splitted_by_headers = md_splitter.split_text(doc.page_content)

    return list_of_docs_splitted_by_headers


def split_by_paragraphs(list_of_docs: List[Document]) -> List[Document]:
    """Splits a list of documents into smaller chunks based on paragraphs."""

    docs_splitted_by_paragraphs: List[Document] = []

    for doc in list_of_docs:
        paragraphs: List[str] = doc.page_content.split("\n")

        for paragraph in paragraphs:
            final_doc = Document(
                page_content=paragraph,
                metadata=doc.metadata,
            )
            docs_splitted_by_paragraphs.append(final_doc)

    print(f"len(docs_splitted_by_paragraphs): {len(docs_splitted_by_paragraphs)}")

    return docs_splitted_by_paragraphs


def splitter(docs: List[Document]) -> List[Document]:
    """Splits documents into chunks based on markdown's header tags."""
    print("\n\n2) CHUNKING DOCUMENTS...")

    chunks = []
    for i, doc in enumerate(docs):
        splits = split_by_headers(doc)
        chunks.extend(splits)
        print(f" 2.{i+1}. chunking document --> {doc.metadata['filename']}")

    return chunks


if __name__ == "__main__":
    loaded_docs = directory_loader(md_dir)

    splits = splitter(loaded_docs)

    splitted_by_paragraphs = split_by_paragraphs(splits)

    for i, doc in enumerate(splitted_by_paragraphs):
        print(
            f"DOC N° {i}:\n\n- METADATA: {doc.metadata}\n\n- CONTENT:\n{doc.page_content}\n\n{'==='*20}\n"
        )
