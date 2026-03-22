import math
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def cosine_similarity(v1, v2):
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

SYSTEM_PROMPT = "You are a helpful travel assistant for Margie's Travel. Only use the provided context to answer the user's questions."

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Margie's Travel RAG Chat", page_icon="✈️", layout="wide")
st.title("✈️ Margie's Travel — RAG Chat")

# ── Sidebar: model configuration ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Configuration")

    # ── Embedding provider & model ────────────────────────────────────────────
    embedding_provider = st.selectbox(
        "Embedding Provider",
        ["Ollama (local)", "OpenAI"],
        index=0,
    )
    use_ollama_embeddings = embedding_provider == "Ollama (local)"

    if use_ollama_embeddings:
        embedding_model = st.selectbox(
            "Embedding Model",
            ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
            index=0,
        )
    else:
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            index=0,
        )

    st.divider()

    # ── Chat provider & model ─────────────────────────────────────────────────
    chat_provider = st.selectbox(
        "Chat Provider",
        ["Ollama (local)", "OpenAI"],
        index=0,
    )
    use_ollama_chat = chat_provider == "Ollama (local)"

    if use_ollama_chat:
        chat_model = st.selectbox(
            "Chat Model",
            ["llama3", "phi3", "mistral", "gemma"],
            index=0,
        )
    else:
        chat_model = st.selectbox(
            "Chat Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )

    st.divider()

    # ── OpenAI API Key (pre-filled from .env) ─────────────────────────────────
    env_key = os.getenv("OPEN_AI_KEY", "")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=env_key,
        type="password",
        help="Loaded from ollama/.env (OPEN_AI_KEY). Required only when using OpenAI.",
    )
    if not use_ollama_embeddings or not use_ollama_chat:
        if not openai_key:
            st.warning("⚠️ OpenAI API key required for selected provider(s).")
        elif env_key and openai_key == env_key:
            st.caption("🔑 Key loaded from `.env`")

    st.divider()
    st.header("📄 Document Corpus")
    for i, doc in enumerate(DOCUMENTS, 1):
        st.markdown(f"**{i}.** {doc}")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.llm_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()

# ── Build clients & vector DB (cached per model selection) ────────────────────
@st.cache_resource(
    show_spinner="Generating embeddings for document corpus...",
)
def build_vector_db(use_ollama_emb: bool, emb_model: str, oai_key: str):
    if use_ollama_emb:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    else:
        client = OpenAI(api_key=oai_key or "your-openai-api-key")

    vector_db = []
    for doc in DOCUMENTS:
        response = client.embeddings.create(input=doc, model=emb_model)
        vector_db.append({"text": doc, "embedding": response.data[0].embedding})
    return client, vector_db


@st.cache_resource
def get_chat_client(use_ollama_chat: bool, oai_key: str):
    if use_ollama_chat:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return OpenAI(api_key=oai_key or "your-openai-api-key")


try:
    embedding_client, vector_db = build_vector_db(
        use_ollama_embeddings, embedding_model, openai_key
    )
    dims = len(vector_db[0]["embedding"])
    st.caption(
        f"Embedding: **{embedding_model}** ({'Ollama' if use_ollama_embeddings else 'OpenAI'}) · "
        f"{dims}d vectors &nbsp;|&nbsp; "
        f"Chat: **{chat_model}** ({'Ollama' if use_ollama_chat else 'OpenAI'})"
    )
except Exception as e:
    st.error(f"❌ Failed to initialise embedding model: {e}")
    st.info(
        "Make sure Ollama is running and the model is pulled:\n"
        f"```\nollama pull {embedding_model}\n```"
    )
    st.stop()

chat_client = get_chat_client(use_ollama_chat, openai_key)

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # display messages (role, content, context)
if "llm_history" not in st.session_state:
    st.session_state.llm_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# ── Render chat history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("context"):
            with st.expander(f"📎 Retrieved context (score: {msg['score']:.4f})"):
                st.info(msg["context"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask Margie's Travel anything…"):
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            # 1. Embed query
            query_response = embedding_client.embeddings.create(
                input=user_input, model=embedding_model
            )
            query_embedding = query_response.data[0].embedding

            # 2. Find best matching document
            best_doc, best_score = None, -1.0
            for item in vector_db:
                score = cosine_similarity(query_embedding, item["embedding"])
                if score > best_score:
                    best_score, best_doc = score, item["text"]

            # 3. Augment prompt
            augmented = f"Context: {best_doc}\n\nUser query: {user_input}"
            current_messages = st.session_state.llm_history.copy()
            current_messages.append({"role": "user", "content": augmented})

        try:
            with st.spinner("Generating response…"):
                # 4. Generate response
                response = chat_client.chat.completions.create(
                    model=chat_model,
                    messages=current_messages,
                )
                answer = response.choices[0].message.content

            st.write(answer)
            with st.expander(f"📎 Retrieved context (score: {best_score:.4f})"):
                st.info(best_doc)

            # Update LLM history with unaugmented user message to keep it clean
            st.session_state.llm_history.append({"role": "user", "content": user_input})
            st.session_state.llm_history.append({"role": "assistant", "content": answer})

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "context": best_doc,
                "score": best_score,
            })

        except Exception as e:
            err = f"Error generating response: {e}"
            st.error(err)
            if use_ollama_chat:
                st.info(f"Make sure Ollama is running and `{chat_model}` is pulled:\n```\nollama pull {chat_model}\n```")
