"""PyMuPDF4llm to load PDF files"""  # !!! FAILS TO LOAD CORRUPT PDF FILES

# pip install -qU langchain-community langchain-core pymupdf pymupdf4llm rich

# GENERAL IMPORTS
import re
from langchain_core.documents import Document
from pathlib import Path
from rich import print as rprint
from rich.progress import track
from typing import List, Dict

# SPECIFIC IMPORTS
from pymupdf4llm import to_markdown
from pymupdf import Document as PyMuDocument

# RICH'S PRINT COLORS
YELLOW = "#fde047"
ORANGE = "#f97316"
RED = "#ef4444"
BLUE = "#3b82f6"
CYAN = "#06b6d4"
EMERALD = "#34d399"
VIOLET = "#a855f7"
PINK = "#ec4899"
GRAY = "#64748b"
WHITE = "#cccccc"
GREEN = "#3fb618"

# PATHS
ROOT_DIR = Path("../../../../../COLEGA DATA")
PDF_DIR = ROOT_DIR / "notificaciones"
PDF_DIR_2 = ROOT_DIR / "MÉTODO DE LA DEMANDA Y SU CONTESTACIÓN" / "CAPS"
PDF_FILE_1 = PDF_DIR / "RES 04-04-2024 - DILIGENCIA PRELIMINAR.pdf"
PDF_FILE_2 = PDF_DIR_2 / "1_EL_CASO_Y_SU_SOLUCIÓN.pdf"


def search_dir(dir_path: Path, file_ext: str) -> List[Dict[str, str]]:
    """FILE'S SEARCH IN A GIVEN DIRECTORY"""
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise ValueError(f"search_dir() => DIRECTORY ({dir_path}) DOESN'T EXIST.")

    if not any(dir_path.iterdir()):
        raise ValueError(f"search_dir() => DIRECTORY ({dir_path}) IS EMPTY.")

    if not file_ext.startswith("."):
        file_ext = f".{file_ext}"

    # SEARCH FOR REQUIRED FILES
    files_info: List[Dict[str, str]] = [
        {"filename": f.name, "filepath": str(f)}
        for f in dir_path.glob(f"*{file_ext}")
        if f.is_file()
    ]

    # CHECK IF FILES WERE FOUND
    if not files_info:
        raise ValueError(
            f"search_dir() => NO FILES WITH EXTENSION ({file_ext}) WERE FOUND IN DIRECTORY ({dir_path})."
        )

    return files_info


def text_cleaner(text: str) -> str:
    """
    CLEANS TEXT BY REPLACING NON-BREAKING SPACES, NORMALIZING SPACES AND NEWLINES,
    AND REMOVING HASH SYMBOLS.
    """

    # FROM NON-BREAKING SPACE CHARACTER TO A REGULAR SPACE
    text = re.sub(r"\xa0", " ", text)
    # FROM MULTIPLE SPACES TO A SINGLE SPACE
    text = re.sub(r" +", " ", text)
    # FROM >=3 - SYMBOLS TO NONE
    text = re.sub(r"-{3,}", "", text)
    # FROM >=3 LINE BREAKS TO DOUBLE LINE BREAKS
    text = re.sub(r"\n{3,}", "\n\n", text)
    # FROM >=2  HASH SYMBOLS TO NONE
    text = re.sub(r"#{2,}", "", text)
    # TRIM LEADING AND TRAILING WHITESPACE
    text = "\n".join([line.strip() for line in text.split("\n\n")])

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


def directory_loader(dir_path: Path, file_ext: str) -> List[List[Document]]:
    """LOADS PDF DOCUMENTS FROM A GIVEN DIRECTORY"""

    # SEARCH IN THE GIVEN DIRECTORY FOR EACH PDF FILE IN IT AND GETS ITS PATH
    files_info: List[Dict[str, str]] = search_dir(dir_path, file_ext)

    # LOADS EACH PDF FILE: FILE --> LIST[DOCUMENT]
    loaded_docs: List[List[Document]] = []
    for f in track(files_info, description="LOADING PDF FILES", total=len(files_info)):
        doc_pages: List[PyMuDocument] = to_markdown(
            f["filepath"], page_chunks=True, show_progress=False
        )

        loaded_file: List[Document] = [
            Document(metadata=page["metadata"], page_content=text_cleaner(page["text"]))
            for page in doc_pages
        ]

        loaded_docs.append(loaded_file)

    return loaded_docs


if __name__ == "__main__":
    # LOADING PDF FILES
    docs = directory_loader(PDF_DIR, "pdf")
    for index, doc in enumerate(docs):
        for page in doc:
            if is_text_corrupt(page.page_content):
                rprint(f"[{RED}]{page.metadata['title']}[/]")
            else:
                rprint(f"[{GREEN}]{page.metadata['title']}[/]")

    for index, doc in enumerate(docs):
        for page in doc:
            rprint(
                f"[bold {BLUE}]> DOC N°:[/] [bold {WHITE}]{index}[/]\n",
                f"[bold {EMERALD}]> FILENAME:[/] [bold {WHITE}]{page.metadata["title"]}[/]\n\n",
                f"[bold {YELLOW}]> CONTENT:[/]\n[{WHITE}]{repr(page.page_content)}[/]",
            )
