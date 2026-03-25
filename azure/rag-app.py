import os
from dotenv import load_dotenv
import glob

# Add references
from openai import OpenAI

def main():
    # Clear the console
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        # Get configuration settings
        load_dotenv()
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        model_deployment = os.getenv("MODEL_DEPLOYMENT")
        api_key = os.getenv("API_KEY")

        # Get an OpenAI client
        client = OpenAI(
            base_url=azure_openai_endpoint,
            api_key=api_key
        )

        # Upload files and create vector store
        print("Creating vector store and uploading files...")
        vector_store = client.vector_stores.create(
            name="travel-brochures"
        )
        file_streams = [open(f, "rb") for f in glob.glob("brochures/*.pdf")]
        if not file_streams:
            print("No PDF files found in the brochures folder!")
            return
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=file_streams
        )
        for f in file_streams:
            f.close()
        print(f"Vector store created with {file_batch.file_counts.completed} files.")

        # Track conversation state
        last_response_id = None

        while True:
            input_text = input('\nEnter a question (or type "quit" to exit): ')
            if input_text.lower() == "quit":
                break
            if len(input_text) == 0:
                print("Please enter a question.")
                continue

            # Get a response grounded exclusively on the uploaded documents
            response = client.responses.create(
                model=model_deployment,
                instructions="""
                You are a travel assistant for Margie's Travel.
                Answer questions ONLY using the information found in the provided travel brochures.
                If the answer cannot be found in the documents, say so clearly — do not use outside knowledge.
                """,
                input=input_text,
                previous_response_id=last_response_id,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [vector_store.id]
                    }
                ]
            )
            print(f"\nAI: {response.output_text}")
            last_response_id = response.id

    except Exception as ex:
        print(ex)

if __name__ == '__main__':
    main()
