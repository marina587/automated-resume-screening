"""
Resume Parser Module
PDF extraction via PyMuPDF; optional spaCy sm refinement for layout cleanup.
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd

SPACY_MODEL = "en_core_web_sm"


class ResumeParser:
    """Extracts text from PDF/DOCX resumes using PyMuPDF + spaCy sm."""

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self.nlp = None
        self.pdf_available = False
        self.docx_available = False
        self._initialize_extractors()

    def _initialize_extractors(self):
        try:
            import fitz  # PyMuPDF
            self._fitz = fitz
            self.pdf_available = True
        except ImportError:
            print("Warning: PyMuPDF not installed. Run: pip install pymupdf")
            self._fitz = None

        try:
            import docx
            self._docx_module = docx
            self.docx_available = True
        except ImportError:
            print("Warning: python-docx not found. Install for DOCX support.")
            self._docx_module = None

        if self.use_spacy:
            try:
                import spacy
                self.nlp = spacy.load(SPACY_MODEL)
            except OSError:
                print(
                    f"Warning: spaCy model '{SPACY_MODEL}' not found. "
                    f"Run: python -m spacy download {SPACY_MODEL}"
                )
                self.nlp = None

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        doc = self._fitz.open(pdf_path)
        try:
            parts = []
            for page in doc:
                parts.append(page.get_text("text"))
            return "\n".join(parts)
        finally:
            doc.close()

    def _extract_with_docx(self, docx_path: str) -> str:
        doc = self._docx_module.Document(docx_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def refine_with_spacy(self, text: str) -> str:
        """Normalize whitespace and rebuild text from spaCy sentences."""
        if not text or self.nlp is None:
            return text.strip() if text else ""

        doc = self.nlp(text[:1_000_000])  # guard very large docs
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return "\n".join(sentences) if sentences else text.strip()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        if not self.pdf_available:
            raise ImportError("PyMuPDF required. Install with: pip install pymupdf")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        text = self._extract_with_pymupdf(pdf_path)
        return self.refine_with_spacy(text.strip())

    def extract_text_from_docx(self, docx_path: str) -> str:
        if not self.docx_available:
            raise ImportError("python-docx not installed.")

        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"File not found: {docx_path}")

        text = self._extract_with_docx(docx_path)
        return self.refine_with_spacy(text.strip())

    def extract_text(self, file_path: str) -> str:
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".pdf":
            return self.extract_text_from_pdf(file_path)
        if file_ext == ".docx":
            return self.extract_text_from_docx(file_path)
        raise ValueError(
            f"Unsupported file format: {file_ext}. Supported: .pdf, .docx"
        )

    def extract_batch(
        self,
        folder_path: str,
        file_extensions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if file_extensions is None:
            file_extensions = [".pdf", ".docx"]

        results = []
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        for ext in file_extensions:
            for file_path in folder.glob(f"*{ext}"):
                try:
                    text = self.extract_text(str(file_path))
                    if text:
                        results.append({
                            "filename": file_path.name,
                            "filepath": str(file_path),
                            "text": text,
                            "file_type": ext[1:].upper(),
                        })
                        print(f"Extracted: {file_path.name}")
                except Exception as e:
                    print(f"Failed to extract {file_path.name}: {e}")

        df = pd.DataFrame(results)
        print(f"Successfully extracted {len(df)} resumes from {folder_path}")
        return df

    def extract_from_uploaded_files(self, uploaded_files: List) -> pd.DataFrame:
        results = []

        for uploaded_file in uploaded_files:
            temp_path = None
            try:
                file_ext = Path(uploaded_file.name).suffix.lower()
                suffix = file_ext or ".pdf"

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = tmp.name

                text = self.extract_text(temp_path)

                if text:
                    results.append({
                        "filename": uploaded_file.name,
                        "text": text,
                        "file_type": file_ext[1:].upper() if file_ext else "UNKNOWN",
                    })

            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {e}")
            finally:
                if temp_path is not None and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        print(f"Warning: could not remove temp file {temp_path}: {e}")

        return pd.DataFrame(results)


def extract_text_from_pdf(pdf_path: str) -> str:
    parser = ResumeParser()
    return parser.extract_text_from_pdf(pdf_path)


if __name__ == "__main__":
    parser = ResumeParser()
    print("Resume Parser (PyMuPDF + spaCy sm)")
    print(f"  PDF (PyMuPDF): {'yes' if parser.pdf_available else 'no'}")
    print(f"  DOCX: {'yes' if parser.docx_available else 'no'}")
    print(f"  spaCy ({SPACY_MODEL}): {'yes' if parser.nlp else 'no'}")
