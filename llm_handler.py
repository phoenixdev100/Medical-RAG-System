"""
LLM Handler Module for Medical RAG System
Handles different LLM integrations (Hugging Face, OpenAI, etc.)
"""

import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LLMHandler:
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 use_gpu: bool = True,
                 max_length: int = 512):
        """
        Initialize LLM Handler
        
        Args:
            model_name: Name of the model to use
            use_gpu: Whether to use GPU if available
            max_length: Maximum response length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        print(f"Initializing LLM: {model_name}")
        print(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            # For medical applications, we'll use a more suitable model
            if "medical" in self.model_name.lower() or "bio" in self.model_name.lower():
                # Use specialized medical models if available
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                # Use a general-purpose model with text generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            print("Falling back to a smaller model...")
            
            # Fallback to a smaller, more reliable model
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model="gpt2",
                    device=0 if self.device == "cuda" else -1
                )
                self.model_name = "gpt2"
                print("Fallback model loaded successfully!")
            except Exception as e2:
                print(f"Error loading fallback model: {str(e2)}")
                raise e2
    
    def generate_response(self, 
                         prompt: str, 
                         context: str = "",
                         temperature: float = 0.7,
                         max_new_tokens: int = 256) -> str:
        """
        Generate response using the LLM
        
        Args:
            prompt: User prompt/question
            context: Relevant context from RAG
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Generated response
        """
        try:
            # Construct the full prompt
            if context:
                full_prompt = f"""Context: {context}

Question: {prompt}

Answer: """
            else:
                full_prompt = f"Question: {prompt}\n\nAnswer: "
            
            # Generate response
            if self.pipeline:
                # Using pipeline
                outputs = self.pipeline(
                    full_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                
                response = outputs[0]['generated_text']
                # Extract only the new part (after the prompt)
                response = response[len(full_prompt):].strip()
                
            else:
                # Using model and tokenizer directly
                inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(full_prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def create_medical_prompt(self, question: str, context: str) -> str:
        """
        Create a medical-specific prompt
        
        Args:
            question: Medical question
            context: Relevant medical context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a medical AI assistant. Based on the provided medical literature context, please answer the following question accurately and professionally.

Medical Context:
{context}

Medical Question: {question}

Please provide a comprehensive answer based on the medical literature provided. If the context doesn't contain sufficient information to answer the question, please state that clearly.

Medical Answer:"""
        
        return prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

class OpenAIHandler:
    """Alternative handler for OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI handler
        
        Args:
            api_key: OpenAI API key (optional, can be set via environment)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("Warning: No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.available = True
        except ImportError:
            print("OpenAI library not installed. Install with: pip install openai")
            self.available = False
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
            self.available = False
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: User prompt
            context: RAG context
            
        Returns:
            Generated response
        """
        if not self.available:
            return "OpenAI API is not available."
        
        try:
            messages = [
                {"role": "system", "content": "You are a medical AI assistant. Provide accurate, professional medical information based on the given context."},
            ]
            
            if context:
                messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"})
            else:
                messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
