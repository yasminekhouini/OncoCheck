"""
OncoCheck Web Interface – Flask Backend
========================================
REST API wrapper for the OncoCheck RAG pipeline.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from OncoCheck import OncoCheckRAG

load_dotenv()
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"]
)

# Initialize RAG system
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "La variable GROQ_API_KEY est absente. "
        "Ajoutez-la dans un fichier .env ou dans vos variables d'environnement."
    )

rag_instance = OncoCheckRAG(api_key=API_KEY)

# Minimum number of vectors expected in a complete index.
# Adjust this after a successful first build by checking the logged count.
EXPECTED_MIN_VECTORS = 100

# Load or build index
print("[OncoCheck] Initializing vector store...")
try:
    count = rag_instance.vector_store.collection.count()
    print(f"[OncoCheck] ChromaDB collection has {count} vectors.")
    if count < EXPECTED_MIN_VECTORS:
        print(f"[OncoCheck] Collection incomplete (< {EXPECTED_MIN_VECTORS} vectors). Rebuilding index...")
        EXCEL_PATH = "cancer_patient_data_sets.xlsx"
        PDF_PATH = "generalites_cancer.pdf"
        rag_instance.build_index(EXCEL_PATH, PDF_PATH)
    else:
        print("[OncoCheck] Existing ChromaDB collection looks complete ✓")
except Exception as e:
    print(f"[OncoCheck] Warning: Could not load/build index: {e}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the web interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """API endpoint for chat queries."""
    try:
        data = request.json
        query = data.get("message", "").strip()
        
        if not query:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Get response from RAG
        response = rag_instance.ask(query, top_k=5, verbose=False)
        
        return jsonify({
            "status": "success",
            "response": response,
            "query": query
        }), 200
    
    except Exception as e:
        print(f"[Error] {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing query: {str(e)}"
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        count = rag_instance.vector_store.collection.count()
        return jsonify({
            "status": "healthy",
            "vectors_loaded": count
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🩺 OncoCheck – Web Interface")
    print("  Starting Flask server on http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)