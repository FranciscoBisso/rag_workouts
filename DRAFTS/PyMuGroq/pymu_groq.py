"""PyMuPDFLoader + LLMImageBlobParser + ChatGroq to load PDF files"""  # !!! FAILS TO LOAD CORRUPT PDF FILES

# pip install -qU langchain-community langchain-groq Pillow pymupdf rich

# GENERAL IMPORTS
import re
from langchain_core.documents import Document
from pathlib import Path
from rich import print
from rich.progress import track
from typing import List, TypedDict

# SPECIFIC IMPORTS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_groq import ChatGroq

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
PDF_FILE_2 = PDF_DIR_2 / "2_EL_RAZONAMIENTO_ARGUMENTATIVO_FORENSE.pdf"

#  LLMIMAGEBLOBPARSER CONFIGURATION
load_dotenv()

LLM_MODEL: ChatGroq = ChatGroq(model="llama-3.2-11b-vision-preview", max_tokens=8192)
PROMPT: str = (
    "You are an assistant tasked with extracting text from images for retrieval."
    + "\n\t1) Extract ALL the text find in the images."
    + "\n\t2) DO NOT exclude any text."
    + "\n\t3) Your answer must be ONLY the extracted text."
    + "\n\t4) DO NOT add to your response more content than the the one find in the file."
    + " For example, DO NOT add explanatory text nor any commets about the task you are performing or anuything."
    + "\nIt is crucial that your response is reliable and limited exclusively to the extracted text."
)


class FileInfo(TypedDict):
    """FILE'S INFO"""

    filename: str
    filepath: str


def files_finder(dir_path: Path | str, file_ext: str = "pdf") -> List[FileInfo]:
    """FILE'S SEARCH IN A GIVEN DIRECTORY"""

    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise ValueError(f"search_dir() => DIRECTORY ({dir_path}) DOESN'T EXIST.")

    if not any(dir_path.iterdir()):
        raise ValueError(f"search_dir() => DIRECTORY ({dir_path}) IS EMPTY.")

    if not file_ext.startswith("."):
        file_ext = f".{file_ext}"

    # SEARCH FOR REQUIRED FILES
    files_info: List[FileInfo] = [
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


def pdf_loader(dir_path: Path, file_ext: str) -> List[List[Document]]:
    """LOADS PDF DOCUMENTS FROM A GIVEN DIRECTORY"""

    # SEARCH IN THE GIVEN DIRECTORY FOR EACH PDF FILE IN IT AND GETS ITS PATH
    loaded_docs: List[List[Document]] = []
    files_info: List[FileInfo] = files_finder(dir_path, file_ext)

    # LOADS EACH PDF FILE: FILE --> LIST[DOCUMENT]
    for f in track(
        files_info,
        description="LOADING PDF FILES",
        total=len(files_info),
    ):
        loaded_file: List[Document] = PyMuPDFLoader(
            file_path=f["filepath"],
            mode="page",
            images_inner_format="text",
            images_parser=LLMImageBlobParser(
                model=LLM_MODEL,
                prompt=PROMPT,
            ),
        ).load()
        for file_page in loaded_file:
            file_page.metadata["title"] = f["filename"]
            content = "".join(
                file_page.page_content.split("\n\n\n\n")[0].split("\n")[1:]
            )
            file_page.page_content = text_cleaner(content)

        loaded_docs.append(loaded_file)

    return loaded_docs


if __name__ == "__main__":
    docs = pdf_loader(PDF_DIR, "pdf")

    for index, doc in enumerate(docs):
        for page in doc:
            if is_text_corrupt(page.page_content):
                print(f"[{RED}]{page.metadata['title']}[/]")
            else:
                print(f"[{GREEN}]{page.metadata['title']}[/]")

    for index, doc in enumerate(docs):
        for page in doc:
            print(
                f"[bold {BLUE}]> DOC N°:[/] [bold {WHITE}]{index}[/]\n",
                f"[bold {EMERALD}]> FILENAME:[/] [bold {WHITE}]{page.metadata["title"]}[/]\n\n",
                f"[bold {YELLOW}]> CONTENT:[/]\n[{WHITE}]{repr(page.page_content)}[/]",
            )
