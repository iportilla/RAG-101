import os
import glob
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Margie's Travel Assistant",
    page_icon="✈️",
    layout="centered",
)

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL_DEPLOYMENT      = os.getenv("MODEL_DEPLOYMENT")
API_KEY               = os.getenv("API_KEY")

# ── OpenAI client (cached so it's created once per session) ──────────────────
@st.cache_resource
def get_client():
    return OpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=API_KEY)

# ── Vector store (cached so files are uploaded only once per session) ─────────
@st.cache_resource
def build_vector_store():
    client = get_client()
    vs = client.vector_stores.create(name="travel-brochures")
    file_streams = [open(f, "rb") for f in glob.glob("brochures/*.pdf")]
    if not file_streams:
        return None, 0
    batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vs.id,
        files=file_streams,
    )
    for f in file_streams:
        f.close()
    return vs.id, batch.file_counts.completed

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content}]  for display
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("✈️ Margie's Travel")
    st.caption("Powered by Azure OpenAI · RAG (file_search)")
    st.divider()

    vector_store_id, file_count = build_vector_store()

    if vector_store_id:
        st.success(f"Vector store ready — {file_count} brochure(s) indexed")
    else:
        st.error("No PDFs found in `brochures/`. Add files and restart.")

    st.divider()
    st.markdown("**Destinations covered:**")
    for pdf in sorted(glob.glob("brochures/*.pdf")):
        name = os.path.splitext(os.path.basename(pdf))[0]
        st.markdown(f"- {name}")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_response_id = None
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Margie's Travel Assistant")
st.caption(
    "Ask anything about our destinations and services. "
    "Answers are grounded exclusively on our travel brochures."
)

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about our travel packages…"):
    if not vector_store_id:
        st.error("Cannot answer — no vector store available.")
        st.stop()

    # Display the user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the Responses API with file_search only
    client = get_client()
    with st.chat_message("assistant"):
        with st.spinner("Searching brochures…"):
            response = client.responses.create(
                model=MODEL_DEPLOYMENT,
                instructions="""
                You are a travel assistant for Margie's Travel.
                Answer questions ONLY using the information found in the provided travel brochures.
                If the answer cannot be found in the documents, say so clearly — do not use outside knowledge.
                """,
                input=prompt,
                previous_response_id=st.session_state.last_response_id,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [vector_store_id],
                    }
                ],
            )
        answer = response.output_text
        st.markdown(answer)

    # Persist state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_response_id = response.id
