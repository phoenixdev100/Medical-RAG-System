"""
Document Processing Module for Medical RAG System
Handles PDF processing, text extraction, and chunking
"""

import os
import fitz  # PyMuPDF
from typing import List, Dict
import tiktoken
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
    except ImportError:
        # Fallback implementation
        class Document:
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}
        
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=200, length_function=len, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function
                self.separators = separators or ["\n\n", "\n", " ", ""]
            
            def split_text(self, text: str) -> List[str]:
                """Simple text splitting implementation"""
                chunks = []
                current_chunk = ""
                
                for char in text:
                    current_chunk += char
                    if len(current_chunk) >= self.chunk_size:
                        chunks.append(current_chunk)
                        # Keep overlap
                        current_chunk = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                return chunks

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                
            doc.close()
            return text
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Process a document and return chunks
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of document chunks
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Extract text based on file type
        if file_path.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        if not text.strip():
            print(f"Warning: No text extracted from {file_path}")
            return []
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
            
        return documents
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all document chunks
        """
        all_documents = []
        supported_extensions = ['.pdf', '.txt', '.md']
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file_path}")
                    
                    try:
                        documents = self.process_document(file_path)
                        all_documents.extend(documents)
                        print(f"Extracted {len(documents)} chunks from {file}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        return all_documents
    
    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
            model: Model name for tokenizer
            
        Returns:
            Number of tokens
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except:
            # Fallback to approximate count
            return len(text.split()) * 1.3
