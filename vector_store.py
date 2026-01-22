"""
Vector Store Module: Responsible for managing and querying the vector database
"""

import os
import json
import pickle
import logging
import fitz
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from config import VECTOR_DB_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, config=None):
        """Initialize vector store"""
        self.config = config or VECTOR_DB_CONFIG
        self.vector_store_path = self.config["vector_store_path"]
        self.embedding_model_name = self.config["embedding_model"]
        self.embedding_dimension = self.config["embedding_dimension"]
        self.distance_metric = self.config["distance_metric"]
        self.top_k_results = self.config["top_k_results"]
        self.use_gpu = self.config.get("use_gpu", True)
        self.batch_size = self.config.get("batch_size", 32)
        
        # Create vector store directory
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Check GPU availability
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Move model to GPU (if configured and available)
        self.embedding_model = self.embedding_model.to(self.device)
        
        # Initialize vector and metadata storage
        self.vectors = {}  # Dictionary to store vectors: {chunk_id: vector}
        self.metadata = {}  # Dictionary to store metadata: {chunk_id: metadata}
        
        # Load existing vector store (if exists)
        self._load_vector_store()
    
    def _load_vector_store(self) -> None:
        """Load existing vector store"""
        vectors_path = os.path.join(self.vector_store_path, "vectors.pkl")
        metadata_path = os.path.join(self.vector_store_path, "metadata.json")
        
        if os.path.exists(vectors_path) and os.path.exists(metadata_path):
            try:
                with open(vectors_path, "rb") as f:
                    self.vectors = pickle.load(f)
                
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                
                logger.info(f"Successfully loaded vector store with {len(self.vectors)} records")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                self.vectors = {}
                self.metadata = {}
    
    def _save_vector_store(self) -> None:
        """Save vector store to disk"""
        try:
            with open(os.path.join(self.vector_store_path, "vectors.pkl"), "wb") as f:
                pickle.dump(self.vectors, f)
            
            with open(os.path.join(self.vector_store_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Vector store saved with {len(self.vectors)} records")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to vector store"""
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} document chunks to vector store")
        
        # Extract all texts for batch embedding
        texts = [doc["text"] for doc in documents]
        
        # Process large amounts of documents in batches to avoid GPU OOM
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            # Encode using GPU
            with torch.no_grad():  # No gradients needed for inference
                batch_embeddings = self.embedding_model.encode(
                    batch_texts, 
                    convert_to_numpy=True,  # Convert to NumPy array for storage
                    show_progress_bar=(len(batch_texts) > 10)  # Show progress bar for large batches
                )
            all_embeddings.extend(batch_embeddings)
        
        # Store embedding vectors and metadata in index
        for i, doc in enumerate(documents):
            chunk_id = doc["chunk_id"]
            self.vectors[chunk_id] = all_embeddings[i]
            
            # Store metadata (excluding text content to save space)
            metadata = doc.copy()
            # Truncate text to prevent excessive metadata size, keeping only first 100 chars as preview
            metadata["text"] = metadata["text"][:100] + "..." if len(metadata["text"]) > 100 else metadata["text"]
            self.metadata[chunk_id] = metadata
        
        # Save updated vector store
        self._save_vector_store()
    
    def update_documents(self, documents: List[Dict]) -> None:
        """Update existing documents or add new documents"""
        if not documents:
            return
        
        # Get document IDs to update
        doc_ids = [doc["chunk_id"] for doc in documents]
        
        # Delete existing documents (if they exist)
        for doc_id in doc_ids:
            if doc_id in self.vectors:
                del self.vectors[doc_id]
            if doc_id in self.metadata:
                del self.metadata[doc_id]
        
        # Add updated documents
        self.add_documents(documents)
        
        logger.info(f"Updated {len(documents)} documents")
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents"""
        if not document_ids:
            return
        
        count = 0
        for doc_id in document_ids:
            if doc_id in self.vectors:
                del self.vectors[doc_id]
                count += 1
            if doc_id in self.metadata:
                del self.metadata[doc_id]
        
        if count > 0:
            self._save_vector_store()
            logger.info(f"Deleted {count} documents")
    
    def delete_by_file_id(self, file_id: str) -> int:
        """Delete all documents associated with a specific file"""
        doc_ids_to_delete = []
        
        for chunk_id, meta in self.metadata.items():
            if meta.get("file_id") == file_id:
                doc_ids_to_delete.append(chunk_id)
        
        self.delete_documents(doc_ids_to_delete)
        
        return len(doc_ids_to_delete)
    
    def search(self, query: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[Dict]:
        """Search for documents most similar to the query"""
        if not self.vectors:
            logger.warning("Vector store is empty, cannot perform search")
            return []
        
        # Use optional parameters or default configuration
        top_k = top_k if top_k is not None else self.top_k_results
        
        # Embed query text (using GPU)
        with torch.no_grad():
            query_vector = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Calculate similarity
        scores = {}
        for chunk_id, vector in self.vectors.items():
            if self.distance_metric == "cosine":
                # Calculate cosine similarity
                score = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            elif self.distance_metric == "euclidean":
                # Calculate Euclidean distance (convert to similarity)
                score = 1 / (1 + np.linalg.norm(query_vector - vector))
            else:  # dot_product
                # Calculate dot product
                score = np.dot(query_vector, vector)
            
            scores[chunk_id] = float(score)  # Convert to standard Python float
        
        # Sort by similarity in descending order
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply similarity threshold filter (if provided)
        if threshold is not None:
            sorted_results = [(chunk_id, score) for chunk_id, score in sorted_results if score >= threshold]
        
        # Get top K results
        top_results = sorted_results[:top_k]
        
        # Return results (including similarity score and full text)
        results = []
        for chunk_id, score in top_results:
            # Get full document content
            with open(self.metadata[chunk_id]["file_path"], "rb") as f:
                doc = fitz.open(stream=f.read())
                start_pos = self.metadata[chunk_id]["start_pos"]
                end_pos = self.metadata[chunk_id]["end_pos"]
                
                # Find the page containing this interval
                full_text = ""
                for page in doc:
                    page_text = page.get_text()
                    full_text += page_text
                    if len(full_text) > end_pos:
                        break
            
            chunk_text = full_text[start_pos:end_pos]
            
            # Build result
            result = {
                **self.metadata[chunk_id],
                "text": chunk_text,  # Use full text
                "score": score
            }
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics of the vector store"""
        unique_files = set()
        for meta in self.metadata.values():
            if "file_id" in meta:
                unique_files.add(meta["file_id"])
        
        return {
            "total_chunks": len(self.vectors),
            "unique_files": len(unique_files),
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "distance_metric": self.distance_metric,
            "device": str(self.device)
        }
    
    def update_config(self, new_config: Dict) -> None:
        """Update configuration"""
        # Check if embedding model needs reloading
        reload_model = (new_config.get("embedding_model") and 
                        new_config["embedding_model"] != self.embedding_model_name)
        
        # Check if device needs changing
        change_device = "use_gpu" in new_config and new_config["use_gpu"] != self.use_gpu
        
        # Update config
        self.config.update(new_config)
        
        # Update member variables
        if "vector_store_path" in new_config:
            self.vector_store_path = new_config["vector_store_path"]
            os.makedirs(self.vector_store_path, exist_ok=True)
        
        if "embedding_dimension" in new_config:
            self.embedding_dimension = new_config["embedding_dimension"]
        
        if "distance_metric" in new_config:
            self.distance_metric = new_config["distance_metric"]
        
        if "top_k_results" in new_config:
            self.top_k_results = new_config["top_k_results"]
        
        if "batch_size" in new_config:
            self.batch_size = new_config["batch_size"]
        
        # If GPU usage setting changed, update device
        if change_device:
            self.use_gpu = new_config["use_gpu"]
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            logger.info(f"Device changed to: {self.device}")
            # Move model to new device
            self.embedding_model = self.embedding_model.to(self.device)
        
        # If embedding model changed, reload model
        if reload_model:
            self.embedding_model_name = new_config["embedding_model"]
            logger.info(f"Reloading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            # Move new model to current device
            self.embedding_model = self.embedding_model.to(self.device)
        
        logger.info(f"Vector store config updated: {self.config}")