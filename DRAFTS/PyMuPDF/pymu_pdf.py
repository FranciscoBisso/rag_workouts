"""PyMuPDF to load PDF files"""  # !!! FAILS TO LOAD CORRUPT PDF FILES

# pip install -qU langchain-community langchain-core pymupdf rich tqdm

# GENERAL IMPORTS
import re
from langchain_core.documents import Document
from pathlib import Path
from rich import print
from typing import List, TypedDict

# SPECIFIC IMPORTS
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser

# RICH'S PRINT COLORS
BLUE = "#3b82f6"
CYAN = "#06b6d4"
EMERALD = "#34d399"
GRAY = "#64748b"
GREEN = "#3fb618"
ORANGE = "#f97316"
PINK = "#ec4899"
RED = "#ef4444"
VIOLET = "#a855f7"
WHITE = "#cccccc"
YELLOW = "#fde047"

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

    # COUNTS ALPHABETIC CHARACTERS, SPACES AND BOM/REPLACEMENT CHARACTERS ("�")
    total_chars = len(text)
    valid_chars = sum(c.isalpha() or c.isspace() for c in text)
    # invalid_chars = sum(1 for c in text if c in "�")

    # IF TOO MANY CORRUPT CHARACTERS OR TOO FEW ALPHABETIC CHARACTERS, MARK AS CORRUPT
    # if (invalid_chars / total_chars) > 0.3:
    if (valid_chars / total_chars) < 0.7:
        return True

    return False


def pdf_loader(dir_path: Path | str) -> List[DocStatus]:
    """LOADS PDF DOCUMENTS FROM A GIVEN DIRECTORY WITH PROGRESS INDICATOR."""

    dir_path = Path(dir_path)
    loaded_documents: List[Document] = GenericLoader(
        blob_loader=FileSystemBlobLoader(
            path=dir_path,
            glob="*.pdf",
            show_progress=True,
        ),
        blob_parser=PyMuPDFParser(
            extract_images=False,
            mode="single",
            pages_delimiter="\n",
        ),
    ).load()

    loaded_files: List[DocStatus] = []
    for loaded_doc in loaded_documents:
        cleaned_text = text_cleaner(loaded_doc.page_content)
        loaded_doc.page_content = cleaned_text

        loaded_files.append(
            DocStatus(is_parsed=False, document=loaded_doc)
            if is_text_corrupt(cleaned_text)
            else DocStatus(
                is_parsed=True,
                document=loaded_doc,
            )
        )

    return loaded_files


if __name__ == "__main__":
    docs = pdf_loader(PDF_DIR)
    for index, doc in enumerate(docs):
        print(
            f"\n[bold {BLUE}]> DOC N°:[/] [bold {WHITE}]{index}[/]",
            f"\n\n[bold {ORANGE}]> PARSED:[/] [bold {WHITE}]{str(doc["is_parsed"]).upper()}[/]",
            f"\n\n[bold {EMERALD}]> FILENAME:[/] [bold {WHITE}]{doc["document"].metadata["title"]}[/]",
            f"\n\n[bold {YELLOW}]> CONTENT:[/]\n[{WHITE}]{doc["document"].page_content}[/]",
            # f"\n\n[bold {YELLOW}]> CONTENT:[/] [{WHITE}]{repr(doc["document"].page_content)}[/]",
            f"[bold {CYAN}]\n\n{'==='*15}[/]",
        )
