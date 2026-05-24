"""
Resume Parser Module
Extracts text from PDF and DOCX resume files.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd


class ResumeParser:
    """Handles text extraction from PDF and DOCX resume files."""
    
    def __init__(self):
        self.pdf_extractor = None
        self.docx_extractor = None
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize text extractors for different file formats."""
        # Try to import pdfminer.six
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            self.pdf_extractor = pdfminer_extract
        except ImportError:
            try:
                import PyPDF2
                self.pdf_extractor = lambda path: self._extract_with_pypdf2(path)
            except ImportError:
                print("Warning: No PDF library found. Install pdfminer.six or PyPDF2 for PDF support.")
                self.pdf_extractor = None
        
        # Try to import python-docx
        try:
            import docx
            self.docx_extractor = lambda path: self._extract_with_docx(path)
        except ImportError:
            print("Warning: python-docx not found. Install it for DOCX support.")
            self.docx_extractor = None
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2."""
        import PyPDF2
        
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def _extract_with_docx(self, docx_path: str) -> str:
        """Extract text using python-docx."""
        import docx
        
        doc = docx.Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        if not self.pdf_extractor:
            raise ImportError("No PDF extractor available. Install pdfminer.six or PyPDF2.")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        try:
            text = self.pdf_extractor(pdf_path)
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Extracted text
        """
        if not self.docx_extractor:
            raise ImportError("python-docx not installed. Install it for DOCX support.")
        
        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"File not found: {docx_path}")
        
        try:
            text = self.docx_extractor(docx_path)
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {docx_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a resume file (auto-detect format).
        
        Args:
            file_path: Path to resume file
            
        Returns:
            Extracted text
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .pdf, .docx")
    
    def extract_batch(self, folder_path: str, file_extensions: List[str] = None) -> pd.DataFrame:
        """
        Extract text from all resume files in a folder.
        
        Args:
            folder_path: Path to folder containing resumes
            file_extensions: List of file extensions to process (default: ['.pdf', '.docx'])
            
        Returns:
            DataFrame with columns: filename, filepath, text, file_type
        """
        if file_extensions is None:
            file_extensions = ['.pdf', '.docx']
        
        results = []
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        for ext in file_extensions:
            for file_path in folder.glob(f'*{ext}'):
                try:
                    text = self.extract_text(str(file_path))
                    if text:  # Only add if text was successfully extracted
                        results.append({
                            'filename': file_path.name,
                            'filepath': str(file_path),
                            'text': text,
                            'file_type': ext[1:].upper()  # Remove dot and uppercase
                        })
                        print(f"Extracted: {file_path.name}")
                except Exception as e:
                    print(f"Failed to extract {file_path.name}: {e}")
        
        df = pd.DataFrame(results)
        print(f"Successfully extracted {len(df)} resumes from {folder_path}")
        return df
    
    def extract_from_uploaded_files(self, uploaded_files: List) -> pd.DataFrame:
        """
        Extract text from uploaded files (for web interface).
        
        Args:
            uploaded_files: List of uploaded file objects (Streamlit UploadFile)
            
        Returns:
            DataFrame with extracted text
        """
        results = []
        
        for uploaded_file in uploaded_files:
            try:
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                # Save temporarily
                temp_path = f"/tmp/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                text = self.extract_text(temp_path)
                
                if text:
                    results.append({
                        'filename': uploaded_file.name,
                        'text': text,
                        'file_type': file_ext[1:].upper() if file_ext else 'UNKNOWN'
                    })
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {e}")
        
        return pd.DataFrame(results)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Convenience function to extract text from a single PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text
    """
    parser = ResumeParser()
    return parser.extract_text_from_pdf(pdf_path)


if __name__ == "__main__":
    # Example usage
    parser = ResumeParser()
    
    # Create test directory with sample data info
    print("Resume Parser initialized successfully")
    print(f"PDF extractor: {'Available' if parser.pdf_extractor else 'Not available'}")
    print(f"DOCX extractor: {'Available' if parser.docx_extractor else 'Not available'}")
    
    # Example of batch extraction (requires actual files)
    # df = parser.extract_batch('data/resumes/')
    # print(df.head())
