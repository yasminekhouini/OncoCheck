"""
OncoCheck – Cancer Risk RAG Chatbot
=====================================
RAG pipeline combining:
  • cancer_patient_data_sets.xlsx  → structured patient knowledge
  • generalites_cancer.pdf         → medical background knowledge
  • FAISS vector store             → semantic retrieval
  • groq API           → response generation
"""

import os
import json
import re
import numpy as np
import pandas as pd
from pypdf import PdfReader
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import pickle

# ─────────────────────────────────────────────
# 1. DOCUMENT LOADING
# ─────────────────────────────────────────────

def load_excel_as_documents(path: str) -> list[dict]:
    """Convert each patient row + statistical summaries into text chunks."""
    df = pd.read_excel(path)
    docs = []

    # ── A) Statistical summary chunks per risk level ──
    for level in ["Low", "Medium", "High"]:
        subset = df[df["Level"] == level]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        means = subset[numeric_cols].mean().round(2)
        text = (
            f"[DATASET SUMMARY – {level.upper()} RISK]\n"
            f"Number of patients: {len(subset)}\n"
            + "\n".join(f"  Average {col}: {val}" for col, val in means.items())
        )
        docs.append({"text": text, "source": f"dataset_summary_{level.lower()}", "type": "summary"})

    # ── B) Risk factor correlation chunk ──
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    level_map = {"Low": 0, "Medium": 1, "High": 2}
    df["_level_num"] = df["Level"].map(level_map)
    correlations = df[numeric_cols + ["_level_num"]].corr()["_level_num"].drop("_level_num").sort_values(ascending=False)
    corr_text = "[RISK FACTOR CORRELATIONS WITH CANCER LEVEL]\n" + "\n".join(
        f"  {col}: {val:.3f}" for col, val in correlations.items()
    )
    docs.append({"text": corr_text, "source": "dataset_correlations", "type": "analysis"})

    # ── C) Individual patient chunks (sampled representatives) ──
    for level in ["Low", "Medium", "High"]:
        subset = df[df["Level"] == level].head(5)
        for _, row in subset.iterrows():
            text = (
                f"[PATIENT PROFILE – {level.upper()} RISK]\n"
                f"Patient ID: {row['Patient Id']}, Age: {row['Age']}, "
                f"Gender: {'Male' if row['Gender'] == 1 else 'Female'}\n"
                f"Risk Factors: Air Pollution={row['Air Pollution']}, "
                f"Alcohol={row['Alcohol use']}, Smoking={row['Smoking']}, "
                f"Genetic Risk={row['Genetic Risk']}, Obesity={row['Obesity']}, "
                f"Dust Allergy={row['Dust Allergy']}, Occupational Hazards={row['OccuPational Hazards']}\n"
                f"Symptoms: Chest Pain={row['Chest Pain']}, Coughing of Blood={row['Coughing of Blood']}, "
                f"Fatigue={row['Fatigue']}, Weight Loss={row['Weight Loss']}, "
                f"Shortness of Breath={row['Shortness of Breath']}, Wheezing={row['Wheezing']}\n"
                f"Lifestyle: Balanced Diet={row['Balanced Diet']}, Passive Smoker={row['Passive Smoker']}\n"
                f"→ Risk Level: {level}"
            )
            docs.append({"text": text, "source": f"patient_{row['Patient Id']}", "type": "patient"})

    # ── D) Scoring rules chunk derived from data ──
    rules_text = """[CANCER RISK SCORING RULES – derived from dataset]
Scores range from 1 (low) to 9 (high) for each factor.

HIGH RISK indicators (score ≥ 6):
  - Smoking, Coughing of Blood, Chest Pain, Shortness of Breath
  - Air Pollution, Alcohol use, Genetic Risk, Occupational Hazards

MEDIUM RISK indicators (score 4–5):
  - Fatigue, Weight Loss, Dust Allergy, Obesity
  - Passive Smoker, Wheezing, Balanced Diet (low score)

LOW RISK profile:
  - Most factor scores ≤ 3
  - No severe symptoms
  - Good balanced diet, low environmental exposure

General rule: if 3+ HIGH indicators score ≥ 6, risk is likely HIGH.
If mostly MEDIUM indicators, risk is MEDIUM. Otherwise LOW."""
    docs.append({"text": rules_text, "source": "scoring_rules", "type": "rules"})

    return docs


def load_pdf_as_documents(path: str, chunk_size: int = 500) -> list[dict]:
    """Extract text from PDF and split into overlapping chunks."""
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Clean up text
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    # Split into chunks with overlap
    words = full_text.split()
    chunks = []
    step = chunk_size - 80  # 80-word overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append({
                "text": f"[MEDICAL KNOWLEDGE – Cancer Généralités]\n{chunk}",
                "source": f"pdf_chunk_{i // step}",
                "type": "medical_knowledge"
            })

    return chunks


# ─────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────

def chunk_documents(docs: list[dict], max_tokens: int = 400) -> list[dict]:
    """Further split very long documents into smaller chunks."""
    chunked = []
    for doc in docs:
        words = doc["text"].split()
        if len(words) <= max_tokens:
            chunked.append(doc)
        else:
            for i in range(0, len(words), max_tokens - 50):
                chunk_text = " ".join(words[i: i + max_tokens])
                chunked.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "type": doc["type"]
                })
    return chunked


# ─────────────────────────────────────────────
# 3. EMBEDDINGS + VECTOR STORE
# ─────────────────────────────────────────────

class VectorStore:
    """
    TF-IDF based vector store with FAISS for fast similarity search.
    Uses TF-IDF embeddings (offline, no HuggingFace needed) normalized
    for cosine similarity, stored in a FAISS IndexFlatIP index.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=4096,      # vocabulary size
            sublinear_tf=True,      # log(1+tf) smoothing
            ngram_range=(1, 2),     # unigrams + bigrams
            min_df=1,
            analyzer="word"
        )
        self.index = None
        self.documents = []

    def build(self, documents: list[dict]):
        """Embed all documents with TF-IDF and build FAISS index."""
        self.documents = documents
        texts = [doc["text"] for doc in documents]
        print(f"[OncoCheck] Computing TF-IDF embeddings for {len(texts)} chunks...")

        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        embeddings = normalize(tfidf_matrix, norm="l2")  # L2 normalize → cosine sim

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"[OncoCheck] FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k most relevant chunks for a query."""
        query_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        query_vec = normalize(query_vec, norm="l2")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                doc = self.documents[idx].copy()
                doc["score"] = float(score)
                results.append(doc)
        return results

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.docs.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        with open(f"{path}.vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"[OncoCheck] Index saved to {path}.*")

    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.docs.json", encoding="utf-8") as f:
            self.documents = json.load(f)
        with open(f"{path}.vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        print(f"[OncoCheck] Index loaded: {self.index.ntotal} vectors")


# ─────────────────────────────────────────────
# 4. LLM GENERATION (Groq API)
# ─────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"   # free-tier model; alternatives: mixtral-8x7b-32768, gemma2-9b-it

def generate_response(query: str, retrieved_chunks: list[dict], api_key: str) -> str:
    """Generate a response using Groq (LLaMA 3) with retrieved context."""
    import urllib.request

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Type: {c['type']} | Score: {c['score']:.3f}]\n{c['text']}"
        for c in retrieved_chunks
    )

    system_prompt = """You are OncoCheck, an intelligent cancer risk assistant chatbot.
You help users understand cancer risk factors, symptoms, and levels based on medical data and research.

IMPORTANT GUIDELINES:
- You are NOT a doctor. Always recommend consulting a healthcare professional for medical decisions.
- Base your answers strictly on the provided context (patient dataset + medical knowledge).
- When assessing risk, explain which factors contribute to the risk level.
- Be clear, empathetic, and informative.
- If asked to predict risk level, ask for the relevant factors if not provided.
- Respond in the same language as the user (French or English).
- For risk prediction, use the scoring rules in the context."""

    user_message = f"""Based on the following retrieved knowledge:

{context}

User question: {query}

Please provide a helpful, accurate response based on the context above."""

    payload = json.dumps({
        "model": GROQ_MODEL,
        "max_tokens": 1000,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "OncoCheck/1.0"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]

    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print("HTTP ERROR:", e.code)
        print("DETAILS:", error_body)
        raise


# ─────────────────────────────────────────────
# 5. FULL RAG PIPELINE
# ─────────────────────────────────────────────

class OncoCheckRAG:
    def __init__(self, api_key: str, index_path: str = "OncoCheck_index"):
        self.api_key = api_key
        self.index_path = index_path
        self.vector_store = VectorStore()

    def build_index(self, excel_path: str, pdf_path: str):
        """Full pipeline: load → chunk → embed → index."""
        print("\n[OncoCheck] ── Step 1: Loading documents ──")
        excel_docs = load_excel_as_documents(excel_path)
        pdf_docs = load_pdf_as_documents(pdf_path)
        all_docs = excel_docs + pdf_docs
        print(f"  Loaded {len(excel_docs)} Excel chunks + {len(pdf_docs)} PDF chunks")

        print("\n[OncoCheck] ── Step 2: Chunking ──")
        chunked = chunk_documents(all_docs)
        print(f"  Total chunks after splitting: {len(chunked)}")

        print("\n[OncoCheck] ── Step 3: Building FAISS index ──")
        self.vector_store.build(chunked)
        self.vector_store.save(self.index_path)

    def load_index(self):
        self.vector_store.load(self.index_path)

    def ask(self, query: str, top_k: int = 5, verbose: bool = False) -> str:
        """RAG query: retrieve relevant chunks → generate response."""
        # Retrieve
        chunks = self.vector_store.search(query, top_k=top_k)

        if verbose:
            print(f"\n[Retrieval] Top {len(chunks)} chunks for: '{query}'")
            for c in chunks:
                print(f"  [{c['score']:.3f}] {c['source']} → {c['text'][:80]}...")

        # Generate
        response = generate_response(query, chunks, self.api_key)
        return response

    def chat(self):
        """Interactive CLI chat loop."""
        print("\n" + "="*60)
        print("  🩺 OncoCheck – Cancer Risk RAG Assistant")
        print("  Type 'quit' to exit | 'verbose' to toggle debug mode")
        print("="*60 + "\n")

        verbose = False
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[OncoCheck] Goodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("[OncoCheck] Goodbye! Take care of your health.")
                break
            if user_input.lower() == "verbose":
                verbose = not verbose
                print(f"[OncoCheck] Verbose mode: {'ON' if verbose else 'OFF'}")
                continue

            print("\nOncoCheck: ", end="", flush=True)
            response = self.ask(user_input, verbose=verbose)
            print(response)
            print()


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv
    
    '''API_KEY = os.environ.get("GROQ_API_KEY", "")'''
    load_dotenv()
    API_KEY = os.getenv("GROQ_API_KEY")
    if not API_KEY:
        raise EnvironmentError(
            "La variable GROQ_API_KEY est absente. "
            "Ajoutez-la dans un fichier .env ou dans vos variables d’environnement."
        )
    EXCEL_PATH = "cancer_patient_data_sets.xlsx"
    PDF_PATH = "generalites_cancer.pdf"
    INDEX_PATH = "OncoCheck_index"

    rag = OncoCheckRAG(api_key=API_KEY, index_path=INDEX_PATH)

    # Build index if it doesn't exist, otherwise load it
    if not os.path.exists(f"{INDEX_PATH}.faiss"):
        rag.build_index("cancer_patient_data_sets.xlsx", "generalites_cancer.pdf")
    else:
        print("[OncoCheck] Loading existing index...")
        rag.load_index()

    # Start chat
    rag.chat()