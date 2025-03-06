"""A generator that uses PyMuPDF to load PDF files"""  # !!! FAILS TO LOAD CORRUPT PDF FILES

# pip install -qU langchain-community langchain-core pymupdf rich tqdm

# GENERAL IMPORTS
import re
from langchain_core.documents import Document
from pathlib import Path
from rich import print
from rich.progress import track
from typing import Generator, List, TypedDict

# SPECIFIC IMPORTS
from langchain_community.document_loaders import PyMuPDFLoader

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


class FileMetadata(TypedDict):
    """FILE'S INFO"""

    filename: str
    filepath: str


class DocStatus(TypedDict):
    """DOCUMENT STATUS"""

    is_parsed: bool
    document: Document


def files_finder(dir_path: Path | str, file_ext: str = "pdf") -> List[FileMetadata]:
    """FILE'S SEARCH IN A GIVEN DIRECTORY"""

    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise ValueError(f"search_dir() => DIRECTORY ({dir_path}) DOESN'T EXIST.")

    if not any(dir_path.iterdir()):
        raise ValueError(f"search_dir() => DIRECTORY ({dir_path}) IS EMPTY.")

    if not file_ext.startswith("."):
        file_ext = f".{file_ext}"

    # SEARCH FOR REQUIRED FILES
    files_info: List[FileMetadata] = [
        FileMetadata(filename=f.name, filepath=str(f))
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


def pdf_loader_generator(dir_path: Path | str) -> Generator[DocStatus, None, None]:
    """
    LOADS PDF DOCUMENTS FROM A GIVEN DIRECTORY:
        1) SEARCHES FOR PDF FILES IN THE SPECIFIED DIRECTORY,
        2) LOADS THEM USING THE PyMuPDF LOADER
        3) CLEANS THE CONTENT OF EACH DOCUMENT.
    AS DOCUMENTS ARE LOADED, THEY ARE GENERATED ONE AT A TIME, ALLOWING FOR
    IMMEDIATE PROCESSING WITHOUT WAITING FOR ALL TO BE LOADED.

    Args:
        dir_path (Path | str): THE PATH OF THE DIRECTORY CONTAINING THE PDF FILES.

    Yields:
        A DICTIONARY CONTAINING TWO KEYS:
            - "is_corrupt": A BOOLEAN INDICATING WHETHER THE DOCUMENT'S CONTENT IS CORRUPT.
            - "document": A LANGCHAIN'S DOCUMENT OBJECT REPRESENTING EACH LOADED AND PROCESSED PDF FILE.
    """

    dir_path = Path(dir_path)

    files_metadata: List[FileMetadata] = files_finder(dir_path, "pdf")

    for f in track(
        files_metadata,
        description=f"[bold {GREEN}]LOADING PDF FILES[/]",
        total=len(files_metadata),
    ):
        loaded_file: Document = PyMuPDFLoader(
            file_path=f["filepath"],
            mode="single",
            pages_delimiter="\n",
        ).load()[0]
        loaded_file.page_content = text_cleaner(loaded_file.page_content)
        loaded_file.metadata["title"] = Path(loaded_file.metadata["source"]).name.split(
            "."
        )[0]

        yield (
            DocStatus(is_parsed=False, document=loaded_file)
            if is_text_corrupt(loaded_file.page_content)
            else DocStatus(is_parsed=True, document=loaded_file)
        )


if __name__ == "__main__":
    docs: Generator[DocStatus, None, None] = pdf_loader_generator(PDF_DIR)

    for index, doc in enumerate(docs):
        print(
            f"\n[bold {BLUE}]> DOC N°:[/] [bold {WHITE}]{index}[/]",
            f"\n\n[bold {ORANGE}]> PARSED:[/] [bold {WHITE}]{str(doc["is_parsed"]).upper()}[/]",
            f"\n\n[bold {EMERALD}]> FILENAME:[/] [bold {WHITE}]{doc["document"].metadata["title"]}[/]",
            # f"\n\n[bold {YELLOW}]> CONTENT:[/]\n[{WHITE}]{doc["document"].page_content}[/]",
            f"\n\n[bold {YELLOW}]> CONTENT:[/] [{WHITE}]{repr(doc["document"].page_content)}[/]",
            f"[bold {CYAN}]\n\n{'==='*15}[/]",
        )
