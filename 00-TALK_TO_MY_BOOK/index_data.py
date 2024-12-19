"""
Module for handling the indexing process of markdown files.
"""

import os
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer


current_dir: str = os.getcwd()
md_dir: str = os.path.join(current_dir, "private_data")
persistent_dir: str = os.path.join(
    current_dir, "database", "actuacion_del_abogado_en_la_causa_judicial"
)
hf_model: str = "intfloat/multilingual-e5-large-instruct"
embeddings_model = HuggingFaceEmbeddings(model_name=hf_model)


def directory_loader(directory_path: str) -> List[Document]:
    """LOADS MARKDOWN DOCUMENTS FROM A GIVEN DIRECTORY."""

    print("1. LOADING DOCUMENTS FROM DIRECTORY...")

    loaded_docs: List[Document] = []

    for root, _, files in os.walk(directory_path):

        md_files = [f for f in files if f.endswith(".md")]

        for i, filename in enumerate(md_files):
            file_path = os.path.join(root, filename)
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

                print(f" 1.{i+1}. loading document -> {filename}")

    loaded_docs.sort(key=lambda doc: doc.metadata["filename"])

    return loaded_docs


def split_by_headers(loaded_docs: List[Document]) -> List[Document]:
    """SPLITS BASED ON MARKDOWN HEADINGS."""

    print("\n2. SPLITTING DOCUMENTS BY HEADERS...")

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

        for chunk in chunks_splitted_by_headers:
            chunk.metadata = {**loaded_doc.metadata, "headers": chunk.metadata}
            docs_splitted_by_headers.append(chunk)

    return docs_splitted_by_headers


def split_by_tokens(list_of_docs: List[Document]) -> List[Document]:
    """SPLITS BASED ON TOKENS"""
    print("\n3. SPLITTING DOCUMENTS BY TOKENS...")

    separators = ["\n\n", "\n", "."]
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=375,
        chunk_overlap=15,
        separators=separators,
    )
    docs_splitted_by_tokens = splitter.split_documents(list_of_docs)

    return docs_splitted_by_tokens


def stringify_metadata_headers(list_of_docs: List[Document]) -> List[Document]:
    """METADATA'S HEADERS FROM DICT TO STRING."""

    for chunk in list_of_docs:
        chunk_headers = chunk.metadata["headers"]

        formatted_headers: str = ""
        for value in chunk_headers:
            formatted_headers += f"{chunk_headers[value]} | "

        chunk.metadata["headers"] = formatted_headers[:-3]

    return list_of_docs


def tokens_per_chunk(list_of_docs: list[Document]) -> str:
    """RETURNS CHUNKS' MAX & MIN LENGTHS IN TOKENS."""

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    token_counts: List[int] = []
    for chunk in list_of_docs:
        full_content = (
            chunk.page_content + chunk.metadata["source"] + chunk.metadata["headers"]
        )
        tokens = tokenizer.encode(full_content)
        token_counts.append(len(tokens))

    min_len = min(token_counts)
    max_len = max(token_counts)

    return f"-> MIN (tokens): {min_len}\n-> MAX (tokens): {max_len}"


def characters_per_chunk(chunks: List[Document]) -> str:
    """RETURNS CHUNKS' MAX & MIN LENGTHS IN CHARACTERS."""

    sizes = [len(chunk.page_content) for chunk in chunks]
    min_len = min(sizes)
    max_len = max(sizes)

    return f"-> MIN (chars): {min_len}\n-> MAX (chars): {max_len}"


def prepare_docs(dir_path: str) -> List[Document]:
    """LOADS FILES, TURNS THEM INTO DOCUMENTS & GETS THEM READY TO BE EMBEDDED."""

    loaded_files: List[Document] = directory_loader(dir_path)
    chunks_splitted_by_md_headers = split_by_headers(loaded_files)
    chunks_splitted_by_tokens = split_by_tokens(chunks_splitted_by_md_headers)
    docs_ready_to_embed = stringify_metadata_headers(chunks_splitted_by_tokens)

    return docs_ready_to_embed


def generate_embeddings(docs_to_embed: List[Document]) -> Chroma:
    """EMBEDS DOCUMENTS INTO A VECTOR STORE."""

    print("\n5. EMBEDDING DOCS...")

    vector_store = Chroma.from_documents(
        documents=docs_to_embed,
        embedding=embeddings_model,
        persist_directory=persistent_dir,
    )

    print("\n6. EMBEDDING PROCESS SUCCESSFULLY FINISHED!")

    return vector_store


if __name__ == "__main__":
    # 0:99||100:131||132:648||649:876||877:1055||1056:1165

    final_docs: List[Document] = prepare_docs(md_dir)

    chunks_characters_sizes: str = characters_per_chunk(final_docs)

    chunks_tokens_sizes: str = tokens_per_chunk(final_docs)

    # embeddings: Chroma = generate_embeddings(final_docs)

    # for index, val in enumerate(final_docs):
    #     print(
    #         f"""DOC N°{index}:\n{val.metadata["headers"]}\n\n{val.page_content}\n\n{'==='*20}\n"""
    #     )

    # print("\n", chunks_characters_sizes, "\n")
    # print(chunks_tokens_sizes)
