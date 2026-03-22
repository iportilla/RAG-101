import math
from openai import OpenAI
import os

def cosine_similarity(v1, v2):
    """
    Measures the cosine of the angle between two multi-dimensional vectors.
    A score of 1.0 means perfectly aligned (identical meaning).
    """
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0.0

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Establish connection to the local Ollama Embedding endpoint
    print("Connecting to local Embedding Model (Ollama)...")
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    embedding_model = "nomic-embed-text" 

    # Our tiny "database" corpus
    documents = [
        "Margie's Travel offers flights to London, Paris, and Rome starting at $399.",
        "We have special 15% discounts for group bookings of 5 or more people.",
        "Our luxury travel packages include hotel stay, breakfast, and private guided tours.",
        "Our customer service team is available 24/7 via the support portal at support@margiestravel.com.",
        "Passport and visa processing services are available for an additional fee."
    ]

    print("\nGenerating vector embeddings for local documents...")
    vector_db = []
    
    for doc in documents:
        try:
            response = client.embeddings.create(input=doc, model=embedding_model)
            vector_db.append({
                "text": doc, 
                "embedding": response.data[0].embedding
            })
        except Exception as e:
            print(f"Error: {e}. Is Ollama running and did you pull {embedding_model}?")
            return
            
    # nomic-embed-text generates vectors with a dimensionality of 768!
    dimensions = len(vector_db[0]["embedding"])
    print(f"Successfully generated vectors for {len(documents)} documents. Each vector has {dimensions} dimensions.")

    while True:
        input_text = input("\nEnter a search query (or 'quit' to exit): ")
        if input_text.lower() == "quit":
            break
        if not input_text.strip():
            continue
            
        # 1. Embed user query into the mathematical space
        query_response = client.embeddings.create(input=input_text, model=embedding_model)
        query_vector = query_response.data[0].embedding

        print("\nCalculating Cosine Similarity distances...")
        
        # 2. Search against Database
        results = []
        for item in vector_db:
            score = cosine_similarity(query_vector, item["embedding"])
            results.append((score, item["text"]))
            
        # 3. Rank Documents
        results.sort(key=lambda x: x[0], reverse=True)
        
        print("\n--- Search Results Ranked by Relevance ---")
        for i, (score, text) in enumerate(results, start=1):
            print(f"[{i}] Score: {score:.4f} | Document: {text}")

if __name__ == '__main__':
    main()
