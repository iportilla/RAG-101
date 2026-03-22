import os
import math
from dotenv import load_dotenv
from openai import OpenAI

# A simple cosine similarity function for local vector search
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0.0

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    load_dotenv()
    
    # Configuration to mix and match OpenAI and local Ollama models.
    # Set to True to use Ollama local models, False to use regular OpenAI models.
    # Ensure you have ollama running locally and the models are pulled.
    # e.g., 'ollama pull nomic-embed-text' and 'ollama pull llama3'
    USE_OLLAMA_FOR_EMBEDDINGS = True
    USE_OLLAMA_FOR_CHAT = True
    
    # Setup OpenAI Client
    # Fallback to empty key if not in .env, so the demo can run with local LLMs 
    # without needing a valid OpenAI key.
    openai_key = os.getenv("OPEN_AI_KEY", "your-openai-api-key")
    openai_client = OpenAI(api_key=openai_key)
    
    # Setup Ollama Client (uses local OpenAI-compatible endpoint)
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    # Define model names to use
    if USE_OLLAMA_FOR_EMBEDDINGS:
        # Example Ollama embedding models: nomic-embed-text, mxbai-embed-large, all-minilm
        embedding_model = "nomic-embed-text" 
        embedding_client = ollama_client
    else:
        embedding_model = "text-embedding-3-small"
        embedding_client = openai_client

    if USE_OLLAMA_FOR_CHAT:
        # Example Ollama chat models: llama3, phi3, mistral
        chat_model = "llama3"
        chat_client = ollama_client
    else:
        chat_model = "gpt-4o-mini" # or gpt-3.5-turbo
        chat_client = openai_client

    # Dummy documents simulating our travel information corpus
    documents = [
        "Margie's Travel offers flights to London, Paris, and Rome starting at $399.",
        "We have special 15% discounts for group bookings of 5 or more people.",
        "Our luxury travel packages include hotel stay, breakfast, and private guided tours.",
        "Our customer service team is available 24/7 via the support portal at support@margiestravel.com.",
        "Passport and visa processing services are available for an additional fee."
    ]

    print(f"Initializing RAG app Demo...")
    print(f"- Embedding Model: {embedding_model} ({'Ollama' if USE_OLLAMA_FOR_EMBEDDINGS else 'OpenAI'})")
    print(f"- Chat Model     : {chat_model}     ({'Ollama' if USE_OLLAMA_FOR_CHAT else 'OpenAI'})")
    
    print("\nGenerating embeddings for local data sources...")
    vector_db = []
    
    for doc in documents:
        try:
            response = embedding_client.embeddings.create(input=doc, model=embedding_model)
            embedding = response.data[0].embedding
            vector_db.append({"text": doc, "embedding": embedding})
        except Exception as e:
            print(f"\nError connecting to embedding model: {e}")
            if USE_OLLAMA_FOR_EMBEDDINGS:
                print("Please ensure your local Ollama server is running and the model is pulled 'ollama pull nomic-embed-text'")
            return
            
    print("Vector database initialized successfully.\n")

    prompt = [
        {"role": "system", "content": "You are a helpful travel assistant for Margie's Travel. Only use the provided context to answer the user's questions."}
    ]

    while True:
        input_text = input("Enter the prompt (or type 'quit' to exit): ")
        if input_text.lower() == "quit":
            break
        if not input_text.strip():
            print("Please enter a prompt.")
            continue
        
        # 1. Embed the user query
        query_response = embedding_client.embeddings.create(input=input_text, model=embedding_model)
        query_embedding = query_response.data[0].embedding
        
        # 2. Search local vector db for the most relevant document using cosine similarity
        best_doc = None
        best_score = -1.0
        for item in vector_db:
            score = cosine_similarity(query_embedding, item["embedding"])
            if score > best_score:
                best_score = score
                best_doc = item["text"]
        
        # 3. Augment the user prompt with context
        augmented_prompt = f"Context: {best_doc}\n\nUser query: {input_text}"
        
        current_messages = prompt.copy()
        current_messages.append({"role": "user", "content": augmented_prompt})

        try:
            # 4. Generate the response
            chat_response = chat_client.chat.completions.create(
                model=chat_model,
                messages=current_messages
            )
            completion = chat_response.choices[0].message.content
            print(f"\n[Retrieved Context (Score: {best_score:.2f})]: {best_doc}")
            print(f"\n[Response]: {completion}\n")
            
            # Keep history context
            prompt.append({"role": "user", "content": input_text})
            prompt.append({"role": "assistant", "content": completion})
            
        except Exception as ex:
            print("\nError generating response:", ex)
            if USE_OLLAMA_FOR_CHAT:
                print("Please ensure your local Ollama server is running and the model is pulled 'ollama pull llama3'")

if __name__ == '__main__':
    main()
