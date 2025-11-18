"""
Setup Script for Medical RAG System
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main setup function"""
    
    print("Setting up Medical RAG System...")
    print("="*50)
    
    # Required packages
    packages = [
        "torch",
        "transformers",
        "sentence-transformers",
        "chromadb",
        "langchain",
        "langchain-community", 
        "pymupdf",
        "tiktoken",
        "datasets",
        "pandas",
        "numpy"
    ]
    
    # Install packages
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
            failed_packages.append(package)
    
    print("\n" + "="*50)
    
    if failed_packages:
        print("Failed to install the following packages:")
        for package in failed_packages:
            print(f"  - {package}")
        print("\nPlease install them manually using:")
        print(f"pip install {' '.join(failed_packages)}")
    else:
        print("All packages installed successfully!")
    
    # Create directories
    directories = [
        "medical_knowledge_db",
        "trained_medical_model",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    print("\nSetup completed!")
    print("\nTo run the Medical RAG System:")
    print("  python medical_rag_system.py")
    print("\nTo train a custom model:")
    print("  python train_model.py")

if __name__ == "__main__":
    main()
