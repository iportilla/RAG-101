import math
import streamlit as st
from openai import OpenAI

def cosine_similarity(v1, v2):
    """
    Measures the cosine of the angle between two multi-dimensional vectors.
    A score of 1.0 means perfectly aligned (identical meaning).
    """
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0.0

DOCUMENTS = [
    "Margie's Travel offers flights to London, Paris, and Rome starting at $399.",
    "We have special 15% discounts for group bookings of 5 or more people.",
    "Our luxury travel packages include hotel stay, breakfast, and private guided tours.",
    "Our customer service team is available 24/7 via the support portal at support@margiestravel.com.",
    "Passport and visa processing services are available for an additional fee."
]

st.set_page_config(page_title="Vector Distance Search Demo", page_icon="🔍", layout="wide")
st.title("🔍 Vector Distance Search Demo")
st.caption("Uses local Ollama embeddings (`nomic-embed-text`) and cosine similarity to rank documents by relevance.")

@st.cache_resource(show_spinner="Connecting to Ollama and generating embeddings...")
def build_vector_db():
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    embedding_model = "nomic-embed-text"
    vector_db = []
    for doc in DOCUMENTS:
        response = client.embeddings.create(input=doc, model=embedding_model)
        vector_db.append({"text": doc, "embedding": response.data[0].embedding})
    return client, embedding_model, vector_db

with st.sidebar:
    st.header("📄 Document Corpus")
    st.write("The following documents are indexed:")
    for i, doc in enumerate(DOCUMENTS, 1):
        st.markdown(f"**{i}.** {doc}")

try:
    client, embedding_model, vector_db = build_vector_db()
    dims = len(vector_db[0]["embedding"])
    st.success(f"✅ Ready — {len(vector_db)} documents indexed · {dims}-dimensional vectors (`{embedding_model}`)")
except Exception as e:
    st.error(f"❌ Could not connect to Ollama: {e}")
    st.info("Make sure Ollama is running and you have pulled `nomic-embed-text`:\n```\nollama pull nomic-embed-text\n```")
    st.stop()

st.divider()

query = st.text_input("Enter a search query:", placeholder="e.g. Do you offer group discounts?")

if query.strip():
    with st.spinner("Embedding query and calculating cosine similarity..."):
        try:
            query_response = client.embeddings.create(input=query, model=embedding_model)
            query_vector = query_response.data[0].embedding

            results = []
            for item in vector_db:
                score = cosine_similarity(query_vector, item["embedding"])
                results.append((score, item["text"]))
            results.sort(key=lambda x: x[0], reverse=True)

            st.subheader("Search Results Ranked by Relevance")
            for rank, (score, text) in enumerate(results, start=1):
                bar_color = "green" if rank == 1 else "blue"
                with st.container(border=True):
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.metric(label=f"Rank #{rank}", value=f"{score:.4f}", delta="top match" if rank == 1 else None)
                    with col2:
                        st.write(text)
                        st.progress(float(score), text=f"Similarity: {score:.4f}")
        except Exception as e:
            st.error(f"Error during search: {e}")
