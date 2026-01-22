"""
Document processing module: Responsible for reading, parsing PDF documents and chunking.
"""

import os
import fitz  # PyMuPDF
import hashlib
from typing import List, Dict, Tuple, Generator
import logging
from config import DOCUMENT_PROCESSOR_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, config=None):
        """Initialize document processor"""
        self.config = config or DOCUMENT_PROCESSOR_CONFIG
        self.chunk_size = self.config["chunk_size"]
        self.chunk_overlap = self.config["chunk_overlap"]
        self.supported_extensions = self.config["supported_extensions"]
        self.max_file_size = self.config["max_file_size_mb"] * 1024 * 1024  # Convert to bytes
    
    def is_file_supported(self, file_path: str) -> bool:
        """Check if the file type is supported"""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_extensions and os.path.getsize(file_path) <= self.max_file_size
    
    def process_pdf(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Process a single PDF file, returning file ID and chunked documents"""
        logger.info(f"Processing file: {file_path}")
        
        # Calculate unique ID for the file
        file_id = self._calculate_file_id(file_path)
        
        # Extract PDF text
        text = self._extract_pdf_text(file_path)
        
        # Split text into chunks
        chunks = list(self._chunk_text(text, file_path, file_id))
        
        logger.info(f"File {file_path} processed, generated {len(chunks)} chunks")
        
        return file_id, chunks
    
    def process_directory(self, directory_path: str) -> Dict[str, List[Dict]]:
        """Process all PDF files in the directory"""
        logger.info(f"Processing directory: {directory_path}")
        
        result = {}
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                if self.is_file_supported(file_path):
                    try:
                        file_id, chunks = self.process_pdf(file_path)
                        result[file_id] = chunks
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                else:
                    logger.info(f"Skipping unsupported file: {file_path}")
        
        return result
    
    def _calculate_file_id(self, file_path: str) -> str:
        """Calculate unique ID based on file path and last modification time"""
        file_stat = os.stat(file_path)
        unique_string = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise
        
        return text
    
    def _chunk_text(self, text: str, file_path: str, file_id: str) -> Generator[Dict, None, None]:
        """Split text into overlapping chunks"""
        if not text:
            return
        
        # Get filename (without path)
        file_name = os.path.basename(file_path)
        
        # Chunk processing
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            if not chunk_text.strip():  # Skip empty chunks
                continue
                
            # Create chunk metadata
            chunk_id = f"{file_id}_{i//self.chunk_size}"
            
            yield {
                "chunk_id": chunk_id,
                "file_id": file_id,
                "file_name": file_name,
                "file_path": file_path,
                "chunk_index": i//self.chunk_size,
                "text": chunk_text,
                "start_pos": i,
                "end_pos": i + len(chunk_text)
            }
    
    def update_config(self, new_config: Dict) -> None:
        """Update configuration parameters"""
        self.config.update(new_config)
        self.chunk_size = self.config["chunk_size"]
        self.chunk_overlap = self.config["chunk_overlap"]
        self.supported_extensions = self.config["supported_extensions"]
        self.max_file_size = self.config["max_file_size_mb"] * 1024 * 1024
        
        logger.info(f"Document processor config updated: {self.config}")