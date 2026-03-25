import os
from dotenv import load_dotenv
import glob
import streamlit as st
from openai import OpenAI

# Initialize page config
st.set_page_config(page_title="Margie's Travel Assistant", page_icon="✈️", layout="wide")

st.title("✈️ Margie's Travel Assistant")
st.markdown("Ask me about travel services and destinations!")

@st.cache_resource(show_spinner="Initializing client and generating vector store...")
def initialize_system():
    load_dotenv()
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("API_KEY")
    model_deployment = os.getenv("MODEL_DEPLOYMENT")

    # Initialize the OpenAI client
    openai_client = OpenAI(
        base_url=azure_openai_endpoint,
        api_key=api_key
    )

    # Create vector store and upload files
    vector_store = openai_client.vector_stores.create(
        name="travel-brochures-streamlit"
    )
    
    pdf_files = glob.glob("brochures/*.pdf")
    if not pdf_files:
        st.error("No PDF files found in the brochures folder!")
        return None, None, None, []
        
    file_streams = [open(f, "rb") for f in pdf_files]
    file_batch = openai_client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams
    )
    for f in file_streams:
        f.close()
        
    return openai_client, vector_store, model_deployment, pdf_files

# Cache ensures vector store is not regenerated on every chat message
client, vector_store, model_deployment, pdf_files = initialize_system()

if client:
    # Sidebar: Show Form Data & Ingested Files
    with st.sidebar:
        st.header("📚 Ingested Knowledge")
        st.success("Vector store initialized successfully!")
        st.write(f"**Total files:** {len(pdf_files)}")
        for f in pdf_files:
            file_name = os.path.basename(f)
            st.markdown(f"- 📄 `{file_name}`")
            
    # Chat Interface Setup
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_response_id" not in st.session_state:
        st.session_state.last_response_id = None

    # Render previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Enter a question about travel plans or brochures..."):
        # Append user message and render
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call the Azure OpenAI Tools API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.responses.create(
                        model=model_deployment,
                        instructions='''
                        You are a travel assistant that provides information on travel services available from Margie's Travel.
                        Answer questions about services offered by Margie's Travel using the provided travel brochures.
                        Search the web for general information about destinations or current travel advice.
                        ''',
                        input=prompt,
                        previous_response_id=st.session_state.last_response_id,
                        tools=[
                            {
                                "type": "file_search",
                                "vector_store_ids": [vector_store.id]
                            },
                            {
                                "type": "web_search_preview"
                            }
                        ]
                    )
                    st.markdown(response.output_text)
                    st.session_state.messages.append({"role": "assistant", "content": response.output_text})
                    st.session_state.last_response_id = response.id
                except Exception as e:
                    st.error(f"Error fetching response: {e}")
