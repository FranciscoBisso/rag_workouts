from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def pdf_loader(paths_list: list[str]) -> list[list[Document]]:
    """LOADS A PDF FILES AND RETURNS A LIST OF DOCUMENT OBJECTS."""

    docs: list[list[Document]] = []
    for pdf_path in paths_list:
        loader = PyPDFLoader(pdf_path)
        loaded_file: list[Document] = loader.load()

        for doc in loaded_file:
            doc.metadata["page"] = doc.metadata["page"] + 1

        docs.append(loaded_file)

    return docs


# if __name__ == "__main__":
# pdf_loader()
