"""
Vector Database Module for Medical RAG System
Handles embeddings and vector storage using ChromaDB
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
try:
    from langchain.schema import Document
except ImportError:
    try:
        from langchain_core.documents import Document
    except ImportError:
        # Fallback implementation
        class Document:
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}
import numpy as np

class VectorDatabase:
    def __init__(self, 
                 db_path: str = "./chroma_db",
                 collection_name: str = "medical_documents",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector database
        
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the collection
            embedding_model: Sentence transformer model name
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to vector database
        
        Args:
            documents: List of Document objects
        """
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents to vector database...")
        
        # Prepare data for ChromaDB
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        print("Documents added successfully!")
    
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def get_relevant_context(self, query: str, n_results: int = 5) -> str:
        """
        Get relevant context for a query
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Concatenated relevant context
        """
        results = self.search(query, n_results)
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant context found."
        
        # Combine relevant documents
        context_parts = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            source = metadata.get('source', 'Unknown')
            
            context_parts.append(f"Source: {source}\nContent: {doc}\n")
        
        return "\n---\n".join(context_parts)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Collection statistics
        """
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "db_path": self.db_path
        }
    
    def clear_collection(self) -> None:
        """
        Clear all documents from collection
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            print("Collection cleared successfully!")
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
