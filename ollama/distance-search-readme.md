# Distance Search Demo

Welcome to the `distance-search-demo.py` companion script! 

While the main `rag-app-demo.py` script showcases the full RAG pipeline (fetching data AND generating an LLM response), this script isolates and highlights purely the **Retrieval** (Vector Search) phase. 

## Why is this important?
Vector search is the heartbeat of RAG. If your retrieval mechanism is bad, the LLM receives the wrong information and generates incorrect, hallucinated answers. This script helps you visualize exactly how a vector database searches and ranks files conceptually without the LLM getting in the way.

## How It Works Under The Hood
1. **Embedding**: Both your documents and your query are sent to the local `nomic-embed-text` model to be converted into high-dimensional mathematical vectors (arrays of 768 floating-point numbers). This maps human language into geometric space.
2. **Cosine Similarity**: We calculate the geometric distance by measuring the cosine of the angle between your query vector and every document vector in our "database". 
    * A score approaching `1.0` means the concepts are pointing in identical directions (highly related meaning).
    * Lower scores indicate less semantic similarity.
3. **Ranking**: The script mathematically ranks the documents from highest similarity score to lowest.

---

## Cosine Similarity — Visual Explainer

### What the Score Actually Means

The score is derived from the **angle** between two vectors in semantic space. Here is how angle maps to score and real-world meaning:

| Angle | Score | Interpretation | Example pair |
|------:|------:|----------------|--------------|
| 0° | 1.00 | Identical meaning | "discount" vs "discount" |
| 15° | 0.97 | Near-synonyms | "group deal" vs "bulk discount" |
| 45° | 0.71 | Related topic | "cheap flights" vs "budget travel" |
| 60° | 0.50 | Loosely related | "vacation" vs "passport" |
| 90° | 0.00 | Unrelated | "travel" vs "algebra" |
| 135° | −0.71 | Opposing topics | "arrival" vs "departure" |
| 180° | −1.00 | Opposite meaning | "cheap" vs "luxury" |

> Most RAG retrieval scores fall in the **0.4 – 0.8** range. A score above **0.7** is a strong match; below **0.3** is essentially noise.

---

### Live Example — Query: `"group deals"`

The embedding model converts the query and every document into a 768-number vector. Then it scores each document:

```
  Query ──► "group deals"

  ┌────┬───────┬──────────────────────────────────┬───────────────────────────────────────────┐
  │ #  │ Score │ Relevance                        │ Document                                  │
  ├────┼───────┼──────────────────────────────────┼───────────────────────────────────────────┤
  │ 1  │ 0.98  │ ████████████████████████  BEST  │ "15% discounts for groups of 5 or more"   │
  │ 2  │ 0.71  │ ██████████████████        HIGH  │ "Flights to London, Paris, Rome at $399"  │
  │ 3  │ 0.55  │ ██████████████            MID   │ "Luxury packages: hotel + guided tours"   │
  │ 4  │ 0.32  │ ████████                  LOW   │ "Passport & visa processing available"    │
  │ 5  │ 0.17  │ ████                      WEAK  │ "24/7 support at support@margies..."      │
  └────┴───────┴──────────────────────────────────┴───────────────────────────────────────────┘

  The top result wins — its text becomes the context injected into the LLM prompt.
```

Note how **"group deals"** matched **"discounts for 5 or more"** with 0.98 — without sharing a single word. The model recognised the same *concept* in two different phrasings.

---

### The Formula

```
                      A · B
  similarity(A,B) = ─────────
                     ‖A‖ × ‖B‖

  where:
    A · B  =  a₁b₁ + a₂b₂ + ... + a₇₆₈b₇₆₈   (dot product across all dimensions)
    ‖A‖    =  sqrt(a₁² + a₂² + ... + a₇₆₈²)   (length / magnitude of vector A)
    ‖B‖    =  sqrt(b₁² + b₂² + ... + b₇₆₈²)   (length / magnitude of vector B)
```

Dividing by the magnitudes **normalises for length** — so a three-word query and a full sentence covering the same topic score equally high.

---

## Word Vectors in Action — Three Diagrams

Each word becomes a 768-number vector. These diagrams compress that to 2D to show the geometry.

---

**Diagram 1 — man vs. woman** `score ≈ 0.95`

```
  ↑
  │   ★ woman
  │  ╱
  │ ╱  } 18° — small angle = high similarity
  │╱
  ●────────────────►
origin            ★ man is close behind (slightly different direction)
```

Gender pairs point in nearly the same direction — not identical, but clearly related.

---

**Diagram 2 — king vs. queen** `score ≈ 0.97`

```
  ↑
  │  ★ queen
  │  ★ king
  │ ╱╱  } 14° — even closer than man/woman!
  │╱╱
  ●────────────────►
origin
```

Royalty pairs cluster even tighter — queen is essentially the feminine form of king in semantic space.

---

**Diagram 3 — The Analogy: king − man + woman ≈ queen**

The arrow from *man → woman* (the gender offset) is the same length and direction as *king → queen*:

```
  ★ queen ══════════════════════════ ★ king
     ↑                                  ↑
  gender                             gender
  offset                             offset
     ↑                                  ↑
  ★ woman ══════════════════════════ ★ man

  ══  royalty offset (same for both pairs)
  ↑   gender offset (same for both pairs)

  king − man + woman  ≈  queen  ✓
```

These four words form a **parallelogram** in semantic space. This is why embedding arithmetic works — meaning has consistent geometric structure.

---

## Running the Demo

### 1. Python Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
# Create the virtual environment (one-time setup)
python -m venv .venv

# Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install all required libraries
pip install -r requirements.txt
```

> You only need to create the venv once. For future sessions just run `source .venv/bin/activate` before running any script.

### 2. Configure environment variables

Copy the sample file and add your OpenAI API key (only required if you switch to OpenAI embeddings — not needed for the default Ollama setup):

```bash
cp .env.sample .env
```

Then open `.env` and replace the placeholder:
```
OPEN_AI_KEY=your_openai_api_key_here
```

> ⚠️ `.env` is listed in `.gitignore` — never commit a file containing a real API key.

### 3. Start Ollama
```bash
ollama serve
ollama pull nomic-embed-text
```

Then execute the demo script in your terminal:
```bash
python distance-search-demo.py
```

### Try these experiments!
When the script prompts you, try typing these conceptually similar (but word-different) phrases:
- `"group deals"` -> Watch how the *15% discount for bookings of 5 or more* sentence ranks #1.
- `"I need to go to europe"` -> Watch how the *London, Paris, Rome* sentence ranks #1.
- `"help me with my travel documents"` -> Watch the *Passport and visa processing* sentence rank #1.

Notice how it successfully matches these phrases even without exact text matching? You're watching the mathematical concept of **Semantic Search** happening directly on your CPU!

---

## Streamlit UI Version

A browser-based UI version of this demo is available in `distance-search-streamlit.py`. It provides the same vector search functionality with an interactive interface.

### Running the UI

Make sure Ollama is running and the embedding model is pulled (same prerequisites as above), then:
```bash
streamlit run distance-search-streamlit.py
```
The app will open automatically at `http://localhost:8501`.

### UI Features

| Feature | Description |
|---|---|
| **Document Corpus Sidebar** | All indexed documents are listed in the sidebar for reference |
| **Status banner** | Confirms the number of indexed documents, vector dimensions, and model name on startup |
| **Search box** | Type any natural-language query and results appear instantly |
| **Ranked result cards** | Each document is shown in a card with its rank, cosine similarity score via `st.metric`, and a colour-coded progress bar so you can visually compare distances at a glance |
| **Top match highlight** | The #1 result is labelled *top match* with a green delta indicator |
| **Cached embeddings** | Document embeddings are generated once on startup (`@st.cache_resource`) — refreshing the page does **not** re-embed the corpus |
| **Error guidance** | If Ollama is unreachable, the app displays a friendly message with the exact `ollama pull` command needed |

### Example Interaction

```
Query: "budget travel to europe"

#1  Score: 0.7231  ████████████████████  Margie's Travel offers flights to London, Paris, and Rome starting at $399.
#2  Score: 0.5814  ████████████░░░░░░░░  Our luxury travel packages include hotel stay, breakfast, and private guided tours.
#3  Score: 0.4902  █████████░░░░░░░░░░░  We have special 15% discounts for group bookings of 5 or more people.
#4  Score: 0.4103  ████████░░░░░░░░░░░░  Passport and visa processing services are available for an additional fee.
#5  Score: 0.3561  ███████░░░░░░░░░░░░░  Our customer service team is available 24/7 via the support portal.
```
