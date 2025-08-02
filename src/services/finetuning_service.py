import json
import logging
from typing import List, Dict, Any
from datetime import datetime
from src.core.config import settings

# Try to import OpenAI fine-tuning client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Hugging Face transformers for local fine-tuning
try:
    from datasets import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class FineTuningService:
    """
    Service for fine-tuning language models for domain adaptation.
    Supports both OpenAI API fine-tuning and local Hugging Face fine-tuning.
    """
    
    def __init__(self):
        self.logger = logger
        self.openai_client = None
        self.fine_tuning_jobs = {}
        
        # Initialize OpenAI client if API key is available
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
    
    def prepare_fine_tuning_dataset(self, qa_pairs: List[Dict[str, str]], 
                                   context: str = None) -> List[Dict[str, str]]:
        """
        Prepare a dataset for fine-tuning from question-answer pairs.
        
        Args:
            qa_pairs: List of dictionaries with 'question' and 'answer' keys
            context: Optional context to include in the prompt
            
        Returns:
            List of formatted training examples
        """
        training_examples = []
        
        for pair in qa_pairs:
            question = pair.get('question', '')
            answer = pair.get('answer', '')
            
            if not question or not answer:
                continue
            
            # Format as prompt-completion pairs
            if context:
                prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
            else:
                prompt = f"Question: {question}\nAnswer:"
            
            training_examples.append({
                "prompt": prompt,
                "completion": answer
            })
        
        return training_examples
    
    def create_fine_tuning_job(self, training_file_path: str, 
                             model: str = "gpt-3.5-turbo-0613",
                             suffix: str = None) -> str:
        """
        Create a fine-tuning job using OpenAI API.
        
        Args:
            training_file_path: Path to the training data file (JSONL format)
            model: Base model to fine-tune
            suffix: Optional suffix for the fine-tuned model name
            
        Returns:
            Fine-tuning job ID
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            # Upload training file
            with open(training_file_path, "rb") as file:
                uploaded_file = self.openai_client.files.create(
                    file=file,
                    purpose="fine-tune"
                )
            
            # Create fine-tuning job
            job = self.openai_client.fine_tuning.jobs.create(
                training_file=uploaded_file.id,
                model=model,
                suffix=suffix
            )
            
            job_id = job.id
            self.fine_tuning_jobs[job_id] = {
                'status': 'created',
                'model': model,
                'training_file': uploaded_file.id,
                'created_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"Created fine-tuning job: {job_id}")
            return job_id
            
        except Exception as e:
            self.logger.error(f"Failed to create fine-tuning job: {e}")
            raise
    
    def get_fine_tuning_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            Job status information
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            job = self.openai_client.fine_tuning.jobs.retrieve(job_id)
            
            status_info = {
                'status': job.status,
                'model': job.fine_tuned_model,
                'created_at': job.created_at,
                'finished_at': job.finished_at,
                'error': getattr(job, 'error', None)
            }
            
            # Update local tracking
            if job_id in self.fine_tuning_jobs:
                self.fine_tuning_jobs[job_id].update(status_info)
            
            return status_info
            
        except Exception as e:
            self.logger.error(f"Failed to get fine-tuning status: {e}")
            raise
    
    def list_fine_tuning_jobs(self) -> List[Dict[str, Any]]:
        """
        List all fine-tuning jobs.
        
        Returns:
            List of job information
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            jobs = self.openai_client.fine_tuning.jobs.list()
            return [{
                'id': job.id,
                'status': job.status,
                'model': job.fine_tuned_model,
                'created_at': job.created_at
            } for job in jobs.data]
            
        except Exception as e:
            self.logger.error(f"Failed to list fine-tuning jobs: {e}")
            raise
    
    def save_training_data(self, training_examples: List[Dict[str, str]], 
                          output_path: str):
        """
        Save training data to a JSONL file.
        
        Args:
            training_examples: List of training examples
            output_path: Path to save the file
        """
        try:
            with open(output_path, 'w') as f:
                for example in training_examples:
                    f.write(json.dumps(example) + '\n')
            
            self.logger.info(f"Saved training data to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save training data: {e}")
            raise
    
    def prepare_local_fine_tuning_dataset(self, training_examples: List[Dict[str, str]]) -> Dataset:
        """
        Prepare a dataset for local Hugging Face fine-tuning.
        
        Args:
            training_examples: List of training examples with 'prompt' and 'completion' keys
            
        Returns:
            Hugging Face Dataset
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ValueError("Transformers library not available")
        
        # Format data for instruction fine-tuning
        formatted_data = []
        for example in training_examples:
            prompt = example.get('prompt', '')
            completion = example.get('completion', '')
            
            # Format as instruction-following example
            formatted_example = {
                'instruction': prompt.split('\nAnswer:')[0],
                'input': '',
                'output': completion
            }
            formatted_data.append(formatted_example)
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        return dataset
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer, 
                         max_length: int = 512) -> Dataset:
        """
        Tokenize a dataset for fine-tuning.
        
        Args:
            dataset: Hugging Face Dataset
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(example):
            # Combine instruction, input, and output
            full_text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            
            # Tokenize
            tokenized = tokenizer(
                full_text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Add labels for training
            tokenized['labels'] = tokenized['input_ids'].clone()
            return tokenized
        
        # Apply tokenization to dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

# Global instance
finetuning_service = FineTuningService()

def create_domain_adaptation_dataset(domain: str, examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Create a domain-specific dataset for fine-tuning.
    
    Args:
        domain: Domain name (e.g., 'insurance', 'legal', 'hr', 'compliance')
        examples: List of domain-specific examples
        
    Returns:
        Formatted training examples
    """
    domain_prompts = {
        'insurance': "You are an insurance policy expert. Answer questions about policy terms, coverage, exclusions, and claims.",
        'legal': "You are a legal document expert. Answer questions about contracts, agreements, liabilities, and legal terms.",
        'hr': "You are an HR policy expert. Answer questions about employment policies, benefits, and workplace procedures.",
        'compliance': "You are a compliance expert. Answer questions about regulations, audit requirements, and compliance procedures.",
        'scientific': "You are a scientific document expert. Answer questions about research findings, methodologies, and scientific principles."
    }
    
    system_prompt = domain_prompts.get(domain, "You are a domain expert. Provide accurate, concise answers.")
    
    training_examples = []
    for example in examples:
        question = example.get('question', '')
        answer = example.get('answer', '')
        context = example.get('context', '')
        
        # Format with domain-specific system prompt
        formatted_example = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Context: {context}\n\nQuestion: {question}"},
                {'role': 'assistant', 'content': answer}
            ]
        }
        training_examples.append(formatted_example)
    
    return training_examples

def prepare_peft_fine_tuning(dataset: Dataset, model_name: str = "gpt2") -> Dataset:
    """
    Prepare dataset for Parameter-Efficient Fine-Tuning (PEFT).
    
    Args:
        dataset: Training dataset
        model_name: Base model name
        
    Returns:
        Prepared dataset for PEFT
    """
    # This is a simplified implementation
    # In practice, you would use libraries like peft from Hugging Face
    
    def add_instruction_template(example):
        # Add instruction template for PEFT
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')
        
        # Format for instruction tuning
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return {'prompt': prompt, 'completion': output_text}
    
    # Apply template to dataset
    peft_dataset = dataset.map(add_instruction_template)
    return peft_dataset

def create_retrieval_aware_training_data(queries: List[str],
                                       contexts: List[str],
                                       answers: List[str]) -> List[Dict[str, str]]:
    """
    Create training data that's aware of the retrieval process.
    
    Args:
        queries: List of user queries
        contexts: List of retrieved contexts
        answers: List of answers
        
    Returns:
        Training examples that include retrieval context
    """
    training_examples = []
    
    for query, context, answer in zip(queries, contexts, answers):
        # Format as retrieval-aware training example
        formatted_example = {
            'messages': [
                {
                    'role': 'system',
                    'content': "You are a retrieval-augmented generation expert. Use the provided context to answer questions accurately."
                },
                {
                    'role': 'user',
                    'content': f"Context: {context}\n\nQuestion: {query}"
                },
                {
                    'role': 'assistant',
                    'content': answer
                }
            ]
        }
        training_examples.append(formatted_example)
    
    return training_examples