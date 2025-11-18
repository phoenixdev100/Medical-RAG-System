"""
Medical RAG System - Main Application
Combines document processing, vector database, and LLM for medical diagnosis assistance
"""

import os
import sys
from typing import List, Dict, Any, Optional
from document_processor import DocumentProcessor
from vector_database import VectorDatabase
from llm_handler import LLMHandler, OpenAIHandler

class MedicalRAGSystem:
    def __init__(self, 
                 db_path: str = "./medical_db",
                 model_name: str = "microsoft/DialoGPT-medium",
                 use_openai: bool = False):
        """
        Initialize Medical RAG System
        
        Args:
            db_path: Path for vector database storage
            model_name: LLM model name
            use_openai: Whether to use OpenAI API
        """
        self.db_path = db_path
        self.model_name = model_name
        self.use_openai = use_openai
        
        print("Initializing Medical RAG System...")
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.vector_db = VectorDatabase(db_path=db_path)
        
        if use_openai:
            self.llm = OpenAIHandler()
        else:
            self.llm = LLMHandler(model_name=model_name)
        
        print("Medical RAG System initialized successfully!")
    
    def load_medical_documents(self, documents_path: str) -> None:
        """
        Load medical documents into the system
        
        Args:
            documents_path: Path to documents (file or directory)
        """
        print(f"Loading medical documents from: {documents_path}")
        
        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Path not found: {documents_path}")
        
        # Process documents
        if os.path.isfile(documents_path):
            documents = self.doc_processor.process_document(documents_path)
        else:
            documents = self.doc_processor.process_directory(documents_path)
        
        if not documents:
            print("No documents were processed!")
            return
        
        print(f"Processed {len(documents)} document chunks")
        
        # Add to vector database
        self.vector_db.add_documents(documents)
        
        # Show statistics
        stats = self.vector_db.get_collection_stats()
        print(f"Database now contains {stats['total_documents']} documents")
    
    def ask_medical_question(self, 
                           question: str, 
                           n_context_docs: int = 5,
                           include_sources: bool = True) -> Dict[str, Any]:
        """
        Ask a medical question and get RAG-enhanced response
        
        Args:
            question: Medical question
            n_context_docs: Number of context documents to retrieve
            include_sources: Whether to include source information
            
        Returns:
            Response with answer and metadata
        """
        print(f"Processing question: {question}")
        
        # Retrieve relevant context
        context = self.vector_db.get_relevant_context(question, n_context_docs)
        
        # Generate response
        if self.use_openai:
            answer = self.llm.generate_response(question, context)
        else:
            # Create medical-specific prompt
            medical_prompt = self.llm.create_medical_prompt(question, context)
            answer = self.llm.generate_response(medical_prompt)
        
        # Prepare response
        response = {
            "question": question,
            "answer": answer,
            "context_used": context if include_sources else None,
            "model_info": self.llm.get_model_info() if hasattr(self.llm, 'get_model_info') else None
        }
        
        return response
    
    def interactive_session(self):
        """
        Start an interactive Q&A session
        """
        print("\n" + "="*60)
        print("Medical RAG System - Interactive Session")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'stats' to see database statistics")
        print("Type 'clear' to clear the database")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("\nEnter your medical question: ").strip()
                
                if question.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                elif question.lower() == 'stats':
                    stats = self.vector_db.get_collection_stats()
                    print(f"\nDatabase Statistics:")
                    print(f"- Total documents: {stats['total_documents']}")
                    print(f"- Collection name: {stats['collection_name']}")
                    print(f"- Database path: {stats['db_path']}")
                    continue
                
                elif question.lower() == 'clear':
                    confirm = input("Are you sure you want to clear the database? (yes/no): ")
                    if confirm.lower() == 'yes':
                        self.vector_db.clear_collection()
                        print("Database cleared!")
                    continue
                
                elif not question:
                    continue
                
                # Process the question
                print("\nProcessing your question...")
                response = self.ask_medical_question(question)
                
                print(f"\nAnswer: {response['answer']}")
                
                if response['context_used']:
                    print(f"\nSources used:")
                    print("-" * 40)
                    print(response['context_used'][:500] + "..." if len(response['context_used']) > 500 else response['context_used'])
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def batch_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions
            
        Returns:
            List of responses
        """
        responses = []
        
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {question}")
            response = self.ask_medical_question(question)
            responses.append(response)
        
        return responses
    
    def export_knowledge_base(self, output_file: str) -> None:
        """
        Export the knowledge base to a file
        
        Args:
            output_file: Output file path
        """
        stats = self.vector_db.get_collection_stats()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Medical RAG System Knowledge Base Export\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total documents: {stats['total_documents']}\n")
            f.write(f"Database path: {stats['db_path']}\n")
            f.write(f"Collection name: {stats['collection_name']}\n\n")
            
            # You could add more detailed export functionality here
        
        print(f"Knowledge base exported to: {output_file}")

def main():
    """Main function to run the Medical RAG System"""
    
    # Configuration
    DB_PATH = "./medical_knowledge_db"
    MODEL_NAME = "microsoft/DialoGPT-medium"  # You can change this to other models
    USE_OPENAI = False  # Set to True if you have OpenAI API key
    
    try:
        # Initialize system
        rag_system = MedicalRAGSystem(
            db_path=DB_PATH,
            model_name=MODEL_NAME,
            use_openai=USE_OPENAI
        )
        
        # Check if we have the medical manual
        medical_manual_path = "manual_medical_diagnosis.pdf"
        
        if os.path.exists(medical_manual_path):
            print(f"Found medical manual: {medical_manual_path}")
            
            # Check if database is empty
            stats = rag_system.vector_db.get_collection_stats()
            if stats['total_documents'] == 0:
                print("Loading medical documents into the system...")
                rag_system.load_medical_documents(medical_manual_path)
            else:
                print(f"Database already contains {stats['total_documents']} documents")
        else:
            print(f"Medical manual not found at: {medical_manual_path}")
            print("You can still use the system, but responses will be limited without medical documents.")
        
        # Start interactive session
        rag_system.interactive_session()
        
    except Exception as e:
        print(f"Error initializing system: {str(e)}")
        print("Please check your dependencies and try again.")

if __name__ == "__main__":
    main()
