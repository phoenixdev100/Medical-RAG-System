# Medical RAG System

A Retrieval-Augmented Generation (RAG) system for medical diagnosis assistance using AI. This system processes medical documents, creates a searchable knowledge base, and provides AI-powered responses to medical questions.

## Features

- **Document Processing**: Extract and process text from PDF medical documents
- **Vector Database**: Store and search medical knowledge using ChromaDB
- **AI Integration**: Support for multiple LLM models (Hugging Face, OpenAI)
- **Interactive Q&A**: Command-line interface for medical questions
- **Model Training**: Fine-tune models on medical data
- **Local Deployment**: Runs entirely on your local system

## Installation

### 1. Quick Setup
Run the setup script to install all dependencies:
```bash
python setup.py
```

### 2. Manual Installation
If you prefer to install manually:
```bash
pip install torch transformers sentence-transformers chromadb langchain langchain-community pymupdf tiktoken datasets pandas numpy
```
```bash
pip install langchain-text-splitters langchain-core
```

## Usage

### 1. Basic Usage
Start the interactive medical Q&A system:
```bash
python medical_rag_system.py
```

This will:
- Load the medical manual PDF (if available)
- Create a vector database of medical knowledge
- Start an interactive session where you can ask medical questions

### 2. Training a Custom Model
To fine-tune a model on your medical data:
```bash
python train_model.py
```

This will:
- Create sample medical Q&A training data
- Fine-tune a language model on medical content
- Save the trained model for use in the RAG system

### 3. Processing Your Own Documents
Place your medical documents (PDF, TXT, MD) in the project directory and the system will automatically process them.

## File Structure

```
capstone project/
├── medical_rag_system.py      # Main application
├── document_processor.py      # PDF and text processing
├── vector_database.py         # ChromaDB integration
├── llm_handler.py            # LLM model handling
├── train_model.py            # Model training script
├── setup.py                  # Installation script
├── README.md                 # This file
├── requirements_basic.txt    # Dependencies list
├── manual_medical_diagnosis.pdf  # Medical reference document
└── medical_knowledge_db/     # Vector database storage
```

## System Components

### 1. Document Processor (`document_processor.py`)
- Extracts text from PDF files using PyMuPDF
- Splits documents into manageable chunks
- Handles multiple document formats

### 2. Vector Database (`vector_database.py`)
- Uses ChromaDB for vector storage
- Sentence-BERT embeddings for semantic search
- Persistent storage and retrieval

### 3. LLM Handler (`llm_handler.py`)
- Support for Hugging Face transformers
- OpenAI API integration (optional)
- Medical-specific prompt engineering

### 4. Main System (`medical_rag_system.py`)
- Orchestrates all components
- Interactive command-line interface
- Batch processing capabilities

### 5. Model Training (`train_model.py`)
- Fine-tune models on medical data
- Create custom Q&A datasets
- Support for various base models

## Configuration

### Model Selection
You can change the LLM model by modifying the `MODEL_NAME` variable in `medical_rag_system.py`:

```python
MODEL_NAME = "microsoft/DialoGPT-medium"  # Default
# MODEL_NAME = "gpt2"  # Smaller, faster
# MODEL_NAME = "microsoft/BioGPT"  # Medical-specific (if available)
```

### OpenAI Integration
To use OpenAI models, set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Then set `USE_OPENAI = True` in the main script.

### Database Configuration
Modify database settings in `vector_database.py`:
- `embedding_model`: Change the sentence transformer model
- `chunk_size`: Adjust document chunk size
- `collection_name`: Change the database collection name

## Interactive Commands

When running the interactive session, you can use these commands:

- **Ask questions**: Simply type your medical question
- **`stats`**: Show database statistics
- **`clear`**: Clear the knowledge database
- **`quit`** or **`exit`**: End the session

## Example Usage

```bash
$ python medical_rag_system.py

Medical RAG System - Interactive Session
========================================

Enter your medical question: What are the symptoms of hypertension?

Processing your question...

Answer: Hypertension, also known as high blood pressure, often presents with no symptoms, which is why it's called the "silent killer." However, some patients may experience headaches, shortness of breath, dizziness, chest pain, or nosebleeds when blood pressure is severely elevated.

Sources used:
Source: manual_medical_diagnosis.pdf
Content: Hypertension is a common cardiovascular condition...
```

## Training Custom Models

### 1. Prepare Training Data
Create a JSON file with medical Q&A pairs:
```json
[
  {
    "question": "What is diabetes?",
    "answer": "Diabetes is a metabolic disorder characterized by high blood sugar levels..."
  }
]
```

### 2. Run Training
```bash
python train_model.py
```

### 3. Use Trained Model
Update the model path in `medical_rag_system.py` to use your trained model.

## Hardware Requirements

### Minimum Requirements
- **CPU**: Modern multi-core processor
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional (CUDA-compatible for faster inference)

### Recommended for Training
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+
- **Storage**: 10GB+ free space

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size in training
   - Use smaller models (e.g., "gpt2" instead of larger models)
   - Close other applications

2. **Model Loading Errors**
   - Check internet connection for model downloads
   - Verify model names are correct
   - Try fallback models

3. **PDF Processing Issues**
   - Ensure PDF files are not corrupted
   - Check file permissions
   - Try converting PDF to text manually

4. **Database Errors**
   - Clear the database directory and restart
   - Check disk space
   - Verify write permissions

### Performance Optimization

1. **Use GPU**: Install CUDA and PyTorch with GPU support
2. **Smaller Models**: Use lightweight models for faster inference
3. **Batch Processing**: Process multiple documents at once
4. **Caching**: The system caches embeddings for faster retrieval

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure compliance with medical data regulations and ethical guidelines when using with real medical data.

## Disclaimer

This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.
