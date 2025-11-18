"""
Training Script for Medical RAG System
Fine-tune models on medical data
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from typing import List, Dict, Any
import pandas as pd
from document_processor import DocumentProcessor
from vector_database import VectorDatabase

class MedicalModelTrainer:
    def __init__(self, 
                 base_model: str = "microsoft/DialoGPT-medium",
                 output_dir: str = "./trained_medical_model"):
        """
        Initialize model trainer
        
        Args:
            base_model: Base model to fine-tune
            output_dir: Directory to save trained model
        """
        self.base_model = base_model
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing trainer with base model: {base_model}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
    
    def prepare_medical_qa_dataset(self, 
                                  questions_file: str = None,
                                  documents_path: str = None) -> Dataset:
        """
        Prepare training dataset from medical Q&A pairs or documents
        
        Args:
            questions_file: Path to JSON file with Q&A pairs
            documents_path: Path to medical documents
            
        Returns:
            Prepared dataset
        """
        training_data = []
        
        # Load from Q&A file if provided
        if questions_file and os.path.exists(questions_file):
            print(f"Loading Q&A data from: {questions_file}")
            with open(questions_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            for item in qa_data:
                if 'question' in item and 'answer' in item:
                    text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                    training_data.append({"text": text})
        
        # Generate training data from documents
        if documents_path and os.path.exists(documents_path):
            print(f"Generating training data from documents: {documents_path}")
            
            # Process documents
            doc_processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
            
            if os.path.isfile(documents_path):
                documents = doc_processor.process_document(documents_path)
            else:
                documents = doc_processor.process_directory(documents_path)
            
            # Create training examples from document chunks
            for doc in documents:
                # Simple approach: use document content as training text
                training_data.append({"text": doc.page_content})
        
        # Create sample medical Q&A if no data provided
        if not training_data:
            print("No training data found. Creating sample medical Q&A...")
            sample_qa = [
                {
                    "question": "What are the symptoms of hypertension?",
                    "answer": "Hypertension often has no symptoms, which is why it's called the 'silent killer'. However, some people may experience headaches, shortness of breath, or nosebleeds."
                },
                {
                    "question": "How is diabetes diagnosed?",
                    "answer": "Diabetes is diagnosed through blood tests including fasting glucose, random glucose, or HbA1c tests. A fasting glucose level of 126 mg/dL or higher indicates diabetes."
                },
                {
                    "question": "What is the treatment for pneumonia?",
                    "answer": "Pneumonia treatment depends on the type and severity. Bacterial pneumonia is typically treated with antibiotics, while viral pneumonia may require supportive care and antiviral medications."
                }
            ]
            
            for item in sample_qa:
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                training_data.append({"text": text})
        
        print(f"Prepared {len(training_data)} training examples")
        
        # Create dataset
        dataset = Dataset.from_pandas(pd.DataFrame(training_data))
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize the training examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def train_model(self, 
                   dataset: Dataset,
                   num_epochs: int = 3,
                   batch_size: int = 4,
                   learning_rate: float = 5e-5,
                   save_steps: int = 500):
        """
        Train the model
        
        Args:
            dataset: Training dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_steps: Steps between saves
        """
        print("Starting model training...")
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=100,
            logging_dir=f"{self.output_dir}/logs",
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training completed! Model saved to: {self.output_dir}")
    
    def create_medical_qa_file(self, output_file: str = "medical_qa_training.json"):
        """
        Create a sample medical Q&A training file
        
        Args:
            output_file: Output file path
        """
        medical_qa = [
            {
                "question": "What are the common symptoms of COVID-19?",
                "answer": "Common symptoms of COVID-19 include fever, cough, shortness of breath, fatigue, body aches, headache, loss of taste or smell, sore throat, and congestion."
            },
            {
                "question": "How is high blood pressure treated?",
                "answer": "High blood pressure is treated through lifestyle changes (diet, exercise, weight management) and medications such as ACE inhibitors, diuretics, beta-blockers, or calcium channel blockers."
            },
            {
                "question": "What is the difference between Type 1 and Type 2 diabetes?",
                "answer": "Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin. Type 2 diabetes occurs when the body becomes resistant to insulin or doesn't produce enough insulin."
            },
            {
                "question": "What are the warning signs of a heart attack?",
                "answer": "Warning signs include chest pain or discomfort, shortness of breath, pain in arms, back, neck, jaw or stomach, cold sweat, nausea, and lightheadedness."
            },
            {
                "question": "How is pneumonia diagnosed?",
                "answer": "Pneumonia is diagnosed through physical examination, chest X-rays, blood tests, sputum tests, and sometimes CT scans or pulse oximetry to measure oxygen levels."
            }
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(medical_qa, f, indent=2, ensure_ascii=False)
        
        print(f"Sample medical Q&A file created: {output_file}")
        return output_file

def main():
    """Main training function"""
    
    # Configuration
    BASE_MODEL = "microsoft/DialoGPT-medium"
    OUTPUT_DIR = "./trained_medical_model"
    DOCUMENTS_PATH = "manual_medical_diagnosis.pdf"
    
    try:
        # Initialize trainer
        trainer = MedicalModelTrainer(
            base_model=BASE_MODEL,
            output_dir=OUTPUT_DIR
        )
        
        # Create sample Q&A file if it doesn't exist
        qa_file = "medical_qa_training.json"
        if not os.path.exists(qa_file):
            trainer.create_medical_qa_file(qa_file)
        
        # Prepare dataset
        print("Preparing training dataset...")
        dataset = trainer.prepare_medical_qa_dataset(
            questions_file=qa_file,
            documents_path=DOCUMENTS_PATH if os.path.exists(DOCUMENTS_PATH) else None
        )
        
        # Train the model
        trainer.train_model(
            dataset=dataset,
            num_epochs=3,
            batch_size=2,  # Small batch size for limited resources
            learning_rate=5e-5
        )
        
        print("Training completed successfully!")
        print(f"Trained model saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Make sure you have sufficient GPU memory or reduce batch size.")

if __name__ == "__main__":
    main()
