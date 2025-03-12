"""
An iterator that uses EasyOCR to load PDF files
* LOADS CORRUPT PDF FILES: not so good with ...Digitally signed by...
"""

# GENERAL IMPORTS
import re
from langchain_core.documents import Document
from pathlib import Path
from rich import print
from rich.progress import track
from typing import Iterator, List, TypedDict, Tuple

# SPECIFIC IMPORTS
from easyocr import Reader
from pdf2image import convert_from_path


# PATHS
CUR_DIR = Path(__file__).cwd()
MODEL_STORE_DIR = CUR_DIR / "model_store"
ROOT_DIR = Path("../../../../../COLEGA DATA")
PDF_DIR = ROOT_DIR / "notificaciones"
PDF_DIR_2 = ROOT_DIR / "MÉTODO DE LA DEMANDA Y SU CONTESTACIÓN" / "CAPS"
PDF_FILE_1 = PDF_DIR / "RES 04-04-2024 - DILIGENCIA PRELIMINAR.pdf"
PDF_FILE_2 = PDF_DIR_2 / "1_EL_CASO_Y_SU_SOLUCIÓN.pdf"


class FileMetadata(TypedDict):
    """FILE'S METADATA"""

    name: str
    path: Path


class DocStatus(TypedDict):
    """DOCUMENT STATUS"""

    is_parsed: bool
    document: Document


def files_finder(dir_path: Path | str, file_type: str = "pdf") -> List[FileMetadata]:
    """
    SEARCHES AND RETRIEVES FILES' METADATA FROM A SPECIFIED DIRECTORY
        ARGS:
            dir_path (Path | str): TARGET DIRECTORY PATH TO SEARCH FILES
            file_type (str): FILE EXTENSION TO FILTER (DEFAULT: 'pdf')

        RETURNS:
            List[FileMetadata]: LIST OF DICTIONARIES CONTAINING FILE INFO:
                - name: FILE'S NAME
                - path: COMPLETE FILE PATH

        RAISES:
            ValueError:
                - IF DIRECTORY DOESN'T EXIST
                - IF DIRECTORY IS EMPTY
                - IF NO FILES WITH SPECIFIED EXTENSION ARE FOUND
    """

    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise ValueError(f"❌ files_finder() => DIRECTORY ({dir_path}) DOESN'T EXIST.")

    if not any(dir_path.iterdir()):
        raise ValueError(f"❌ files_finder() => DIRECTORY ({dir_path}) IS EMPTY.")

    if not file_type.startswith("."):
        file_type = f".{file_type}"

    # SEARCH FOR REQUIRED FILES
    files_metadata: List[FileMetadata] = [
        FileMetadata(name=f.name, path=f)
        for f in track(
            dir_path.glob(f"*{file_type}"),
            description="[bold cyan]🔍 SEARCHING FILES[/]",
            total=len(list(dir_path.glob(f"*{file_type}"))),
        )
        if f.is_file()
    ]

    # CHECK IF FILES WERE FOUND
    if not files_metadata:
        raise ValueError(
            f"❌ files_finder() => NO FILES OF TYPE ({file_type}) WERE FOUND IN DIRECTORY ({dir_path})."
        )

    return files_metadata


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


def pdf_loader(dir_path: Path | str, file_type: str = "pdf") -> Iterator[DocStatus]:
    """
    LOADS PDF DOCUMENTS FROM A GIVEN DIRECTORY:
        1) SEARCHES FOR PDF FILES IN THE SPECIFIED DIRECTORY,
        2) LOADS THEM USING EasyOCR
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

    reader = Reader(["es", "en"], model_storage_directory=MODEL_STORE_DIR)

    files_metadata: List[FileMetadata] = files_finder(dir_path, file_type)

    for f in track(
        files_metadata,
        description="📂 [bold yellow]LOADING PDF FILES[/]",
        total=len(files_metadata),
    ):
        f_pages_imgs = convert_from_path(f["path"], fmt="jpeg")

        pages_text: List[str] = []
        for pag in f_pages_imgs:
            results: List[Tuple[List[int], str, float]] = reader.readtext(pag)
            # EXTRACT TEXT FROM RESULTS
            page_extracted_text = " ".join([tupl[1] for tupl in results])
            pages_text.append(page_extracted_text)

        cleaned_text: str = text_cleaner("\n".join(pages_text))

        yield (
            DocStatus(
                is_parsed=False,
                document=Document(metadata=f, page_content=cleaned_text),
            )
            if is_text_corrupt(cleaned_text)
            else DocStatus(
                is_parsed=True,
                document=Document(metadata=f, page_content=cleaned_text),
            )
        )


if __name__ == "__main__":
    easy_docs: Iterator[DocStatus] = pdf_loader(PDF_DIR)
    for index, doc in enumerate(easy_docs):
        print(
            f"\n[bold sky_blue2]> DOC N°:[/] [bold grey93]{index}[/]",
            f"\n\n[bold light_coral]> PARSED:[/] [bold grey93]{str(doc['is_parsed']).upper()}[/]",
            f"\n\n[bold sea_green1]> FILENAME:[/] [bold grey93]{doc['document'].metadata['name']}[/]",
            f"\n\n[bold yellow]> CONTENT:[/]\n[grey93]{repr(doc['document'].page_content)}[/]",
            f"\n\n[bold cyan]{'===' * 15}[/]",
        )
