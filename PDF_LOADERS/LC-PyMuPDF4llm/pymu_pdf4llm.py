"""
PyMuPDF4LLM to load PDF files
!!! FAILS TO LOAD CORRUPT PDF FILES
"""

# GENERAL IMPORTS
import re
from langchain_core.documents import Document
from pathlib import Path
from rich import print
from typing import List, TypedDict

# SPECIFIC IMPORTS
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_pymupdf4llm import PyMuPDF4LLMParser


# PATHS
CUR_DIR = Path(__file__).cwd()
ROOT_DIR = Path("../../../../../COLEGA DATA")
PDF_DIR = ROOT_DIR / "notificaciones"
PDF_DIR_2 = ROOT_DIR / "MÉTODO DE LA DEMANDA Y SU CONTESTACIÓN" / "CAPS"
PDF_FILE_1 = PDF_DIR / "RES 04-04-2024 - DILIGENCIA PRELIMINAR.pdf"
PDF_FILE_2 = PDF_DIR_2 / "1_EL_CASO_Y_SU_SOLUCIÓN.pdf"


class DocStatus(TypedDict):
    """DOCUMENT STATUS"""

    is_parsed: bool
    document: Document


def text_cleaner(text: str) -> str:
    """
    CLEANS TEXT BY REPLACING NON-BREAKING SPACES & NORMALIZING SPACES AND NEWLINES.
    """

    # FROM NON-BREAKING SPACE CHARACTER TO A REGULAR SPACE
    text = re.sub(r"\xa0", " ", text)
    # FROM MULTIPLE SPACES TO A SINGLE SPACE
    text = re.sub(r" {2,}", " ", text)
    # FROM >=3 LINE BREAKS TO DOUBLE LINE BREAKS
    text = re.sub(r"\n{3,}", "\n\n", text)
    # TRIM LEADING AND TRAILING WHITESPACE
    text = "\n\n".join(
        [double_line_break.strip() for double_line_break in text.split("\n\n")]
    )

    text = text.strip()

    return text


def is_text_corrupt(text) -> bool:
    """VERIFIES IF THE EXTRACTED TEXT CONTAINS CORRUPT CHARACTERS OR ITS ENCODED INCORRECTLY."""
    if not text.strip():
        return True

    # COUNTS ALPHABETIC CHARACTERS & SPACES
    total_chars = len(text)
    valid_chars = sum(c.isalpha() or c.isspace() for c in text)

    # IF TOO FEW ALPHABETIC CHARACTERS, MARK AS CORRUPT
    if (valid_chars / total_chars) < 0.7:
        return True

    return False


def pdf_loader(dir_path: Path | str) -> List[DocStatus]:
    """LOADS PDF DOCUMENTS FROM A GIVEN DIRECTORY"""
    dir_path = Path(dir_path)

    documents = GenericLoader(
        blob_loader=FileSystemBlobLoader(
            path=dir_path,
            glob="*.pdf",
            show_progress=True,
        ),
        blob_parser=PyMuPDF4LLMParser(
            mode="single",
            pages_delimiter="\n",
        ),
    ).load()

    loaded_docs: List[DocStatus] = []
    for d in documents:
        d.page_content = text_cleaner(d.page_content)
        loaded_docs.append(
            DocStatus(is_parsed=False, document=d)
            if is_text_corrupt(d.page_content)
            else DocStatus(is_parsed=True, document=d)
        )

    return loaded_docs


if __name__ == "__main__":
    docs: List[DocStatus] = pdf_loader(PDF_DIR)
    for index, doc in enumerate(docs):
        print(
            f"\n[bold sky_blue2]> DOC N°:[/] [bold grey93]{index}[/]",
            f"\n\n[bold light_coral]> PARSED:[/] [bold grey93]{str(doc['is_parsed']).upper()}[/]",
            f"\n\n[bold sea_green1]> FILENAME:[/] [bold grey93]{doc['document'].metadata['title']}[/]",
            f"\n\n[bold yellow]> CONTENT:[/]\n[grey93]{repr(doc['document'].page_content)}[/]",
            f"\n\n[bold cyan]{'===' * 15}[/]",
        )
