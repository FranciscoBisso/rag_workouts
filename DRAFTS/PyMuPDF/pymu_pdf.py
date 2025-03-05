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


class FileInfo(TypedDict):
    """FILE'S INFO"""

    filename: str
    filepath: str


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
    text = "\n".join(
        [double_line_break.strip() for double_line_break in text.split("\n")]
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


def pdf_loader(dir_path: Path | str) -> List[Document]:
    """LOADS PDF DOCUMENTS FROM A GIVEN DIRECTORY WITH PROGRESS INDICATOR."""

    loaded_files: List[Document] = GenericLoader(
        blob_loader=FileSystemBlobLoader(
            path=dir_path,
            glob="*.pdf",
            show_progress=True,
        ),
        blob_parser=PyMuPDFParser(
            extract_images=False,
            mode="single",
        ),
    ).load()

    return loaded_files


if __name__ == "__main__":
    docs = pdf_loader(PDF_DIR)

    for index, doc in enumerate(docs):
        if is_text_corrupt(doc.page_content):
            print(f"[{RED}]{doc.metadata['title']}[/]")
        else:
            print(f"[{GREEN}]{doc.metadata['title']}[/]")

    for i, doc in enumerate(docs):
        # doc.page_content = text_cleaner(doc.page_content)
        print(
            f"[bold {BLUE}]> DOC N°:[/] [bold {WHITE}]{i}[/]\n",
            f"[bold {EMERALD}]> FILENAME:[/] [bold {WHITE}]{doc.metadata['title']}[/]\n\n",
            f"[bold {YELLOW}]> CONTENT:[/]\n[{WHITE}]{doc.page_content[:2000]}[/]",
            # f"[bold {YELLOW}]> CONTENT:[/]\n[{WHITE}]{repr(doc.page_content)}[/]",
            f"\n\n{'==='*15}\n",
        )
