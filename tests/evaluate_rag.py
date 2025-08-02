import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
import os
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Import the actual services from our application
from src.services import ingestion_service, rag_workflow_service

# Load environment variables from .env file
load_dotenv()

async def run_real_rag_pipeline(question: str, document_url: str, document_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs the actual RAG pipeline for a single question to be evaluated.
    """
    print(f"Running real pipeline for question: '{question}'")
    
    # The document is already chunked, so we just need to run the question processing part.
    # We use process_single_question which assumes the document is already indexed.
    result = await rag_workflow_service.process_single_question(
        question=question,
        document_url=document_url,
        document_chunks=document_chunks,
        question_index=0 # Index is just for logging here
    )
    
    # The result is a dictionary with "answer" and "contexts"
    return {
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"]
    }

def get_evaluation_dataset() -> Tuple[Dataset, str]:
    """
    Creates and returns the golden dataset for evaluation from a payload.
    """
    # Using payload1.json for this evaluation
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    questions = [
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?",
        "How does the policy define a 'Hospital'?",
    ]
    # NOTE: These ground truths should be manually verified for accuracy.
    # They are placeholders for a real evaluation.
    ground_truths = [
        "The waiting period for pre-existing diseases (PED) is 48 months of continuous coverage from the date of inception of the policy.",
        "Yes, for Plan A, the sub-limit for room rent is up to 1% of the Sum Insured per day and for ICU charges it is up to 2% of the Sum Insured per day.",
        "A hospital is defined as an institution with at least 15 in-patient beds, an operating theatre, and qualified medical staff available 24/7.",
    ]
    
    data = {
        "question": questions,
        "ground_truth": ground_truths,
    }
    return Dataset.from_dict(data), document_url

async def main():
    """
    Main function to run the RAG evaluation.
    """
    print("Starting RAG pipeline evaluation with real components...")

    # 1. Load the evaluation dataset and the target document URL
    eval_dataset, document_url = get_evaluation_dataset()
    
    # 2. Pre-process the document once: download and chunk it.
    print(f"Processing document: {document_url}")
    document_chunks = await ingestion_service.process_and_extract(document_url)
    
    # 3. Index the document in Pinecone once before running questions.
    print(f"Indexing {len(document_chunks)} chunks in Pinecone...")
    await rag_workflow_service.embedding_service.embed_and_upsert_chunks(document_url, document_chunks)

    # 4. Run the RAG pipeline for each question in the dataset
    results = []
    for item in eval_dataset:
        rag_output = await run_real_rag_pipeline(item['question'], document_url, document_chunks)
        # Add the ground_truth to the results for ragas
        rag_output['ground_truth'] = item['ground_truth']
        results.append(rag_output)

    # 5. Prepare the results for the ragas evaluator
    result_dataset = Dataset.from_list(results)

    # 6. Define the metrics for evaluation
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # 7. Run the evaluation
    print("Running Ragas evaluation...")
    score = evaluate(
        dataset=result_dataset,
        metrics=metrics,
    )

    print("Evaluation complete.")
    print(score)
    df = score.to_pandas()
    print("\nEvaluation Results DataFrame:")
    print(df.to_string())

if __name__ == "__main__":
    asyncio.run(main())