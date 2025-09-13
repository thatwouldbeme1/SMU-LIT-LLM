import os
import json
import chromadb
import google.generativeai as genai
from tqdm import tqdm
import time

# --- Configuration ---
# IMPORTANT: Go to https://aistudio.google.com/app/apikey to get your API key
GOOGLE_API_KEY = "API_KEY"
DATA_DIRECTORY = "data"  # Create this folder and put all 580 JSON files inside it
COLLECTION_NAME = "jus_mundi"
EMBEDDING_MODEL = "models/text-embedding-004"
CHUNK_SIZE = 1500  # Legal text can be dense, slightly larger chunk size is okay
CHUNK_OVERLAP = 250

# --- Setup ---
if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY" or not GOOGLE_API_KEY:
    print("ERROR: Please set your GOOGLE_API_KEY in the ingest.py script.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

# --- NEW RECURSIVE TEXT EXTRACTION LOGIC ---

def _extract_text_recursively(data, texts):
    """Helper function to recursively find all long strings in a JSON object."""
    if isinstance(data, dict):
        for value in data.values():
            _extract_text_recursively(value, texts)
    elif isinstance(data, list):
        for item in data:
            _extract_text_recursively(item, texts)
    elif isinstance(data, str) and len(data) > 150: # Increased threshold to filter out metadata better
        texts.append(data)

def get_text_from_json(file_path):
    """
    Loads a JSON file and uses a recursive helper to extract all meaningful text.
    This is far more robust for complex, nested JSON structures.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        extracted_texts = []
        _extract_text_recursively(data, extracted_texts)
        
        # Add the document title for better context if available
        title = data.get("Title", "Unknown Document")
        
        return f"DOCUMENT TITLE: {title}\n\n" + "\n\n---\n\n".join(extracted_texts)
    except Exception as e:
        print(f"Error reading or parsing {file_path}: {e}")
        return None

# --- Main Logic ---

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def main():
    """Main function to ingest data into ChromaDB."""
    print("--- Starting Data Ingestion (v2 - Robust Parsing) ---")

    client = chromadb.PersistentClient(path="./chroma_db")
    
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting for a fresh start.")
        client.delete_collection(name=COLLECTION_NAME)
    
    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"Created new collection: '{COLLECTION_NAME}'")

    if not os.path.isdir(DATA_DIRECTORY):
        print(f"ERROR: The directory '{DATA_DIRECTORY}' was not found.")
        return

    json_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files to process.")

    all_chunks = []
    all_metadatas = []

    print("Phase 1/3: Extracting and chunking text...")
    for filename in tqdm(json_files, desc="Processing files"):
        file_path = os.path.join(DATA_DIRECTORY, filename)
        document_text = get_text_from_json(file_path)

        if document_text:
            chunks = split_text_into_chunks(document_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"source": filename, "chunk_id": i})

    print(f"Generated a total of {len(all_chunks)} text chunks.")
    if not all_chunks:
        print("ERROR: No text chunks were generated. The extraction logic might still need tuning.")
        return

    batch_size = 100
    print(f"\nPhase 2/3: Generating embeddings in batches of {batch_size}...")

    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        batch_chunks = all_chunks[i:i + batch_size]
        
        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch_chunks,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = response['embedding']
            batch_ids = [f"{all_metadatas[i+j]['source']}_{all_metadatas[i+j]['chunk_id']}" for j in range(len(batch_chunks))]
            batch_metadatas = all_metadatas[i:i + batch_size]

            collection.add(
                embeddings=embeddings,
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
        except Exception as e:
            print(f"\nAn error occurred during batch {i//batch_size + 1}: {e}")
            print("This might be a rate limit issue. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            try:
                response = genai.embed_content(model=EMBEDDING_MODEL, content=batch_chunks, task_type="RETRIEVAL_DOCUMENT")
                embeddings = response['embedding']
                collection.add(embeddings=embeddings, documents=batch_chunks, metadatas=batch_metadatas, ids=batch_ids)
                print("Retry successful.")
            except Exception as retry_e:
                print(f"Retry failed: {retry_e}. Skipping this batch.")
    
    print("\nPhase 3/3: Finalizing...")
    print("\n--- Data Ingestion Complete! ---")
    print(f"Successfully added {collection.count()} documents to the '{COLLECTION_NAME}' collection.")
    print("You are now ready to build the core RAG logic.")

if __name__ == "__main__":

    main()
