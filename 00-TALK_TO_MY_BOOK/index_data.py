"""
Module for handling the indexing process of the markdown files.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

current_dir = os.getcwd()
# md_dir = os.path.join(os.getcwd(), "private_data", "md")
md_dir = os.path.join(os.getcwd(), "private_data")


def directory_loader(directory_path: str) -> List[Document]:
    """Loads markdown documents from a given directory."""
    print("Loading documents from directory...".upper())

    documents = []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".md"):
                file_path = os.path.join(root, filename)
                directory_name = os.path.basename(root)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    document_schema = Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "directory": directory_name,
                            "filename": filename.replace(".md", ""),
                        },
                    )
                    documents.append(document_schema)
                    print(f"- loading document --> {filename}")

    documents.sort(key=lambda doc: doc.metadata["filename"])

    return documents


def splitter_by_headers(doc: Document) -> List[Document]:
    """Splits a single document into chunks based on markdown's header tags."""
    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=True
    )

    splits = splitter.split_text(doc.page_content)

    return splits


def chunker_by_headers(docs: List[Document]) -> List[Document]:
    """Splits documents into chunks based on markdown's header tags."""
    print("\n\nChunking documents...".upper())
    chunks = []
    for doc in docs:
        splits = splitter_by_headers(doc)
        chunks.extend(splits)
        print(f"- chunking document --> {doc.metadata['filename']}")

    return chunks


if __name__ == "__main__":
    loaded_docs = directory_loader(md_dir)
    # for i, document in enumerate(loaded_docs):
    #     print(f"\nDocument N° {i}:\n- Metadata:\n{document.metadata}\n")

    # chunks_split_by_headers = splitter_by_headers(loaded_docs[4])
    # for i, chunk in enumerate(chunks_split_by_headers):
    #     print(
    #         f"{'==='*15}\nChunk N° {i}:\nlen: {len(chunk.page_content)}\n\n-Metadata: {chunk.metadata}\n\n{chunk.page_content[:337]}\n{'==='*20}\n\n"
    #     )
    chunker = chunker_by_headers(loaded_docs)
    print("\n\nLEN(CHUNKER): ", len(chunker))
    print("TYPE(CHUNKER): ", type(chunker))
