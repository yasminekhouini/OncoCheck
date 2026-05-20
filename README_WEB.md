# OncoCheck Web Interface

A beautiful, interactive web interface for the OncoCheck cancer risk RAG chatbot.

## 🚀 Features

- **Modern Web UI**: Clean, responsive chat interface built with HTML, CSS, and JavaScript
- **Real-time Chat**: Instant responses from the OncoCheck RAG pipeline
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Professional Styling**: Gradient backgrounds, smooth animations, and intuitive layout
- **Health Check**: API endpoint to verify system status
- **Error Handling**: Graceful error messages for connection issues

## 📋 Prerequisites

- Python 3.8+
- Flask and dependencies (see `requirements.txt`)
- GROQ API key (set in `.env` file)
- Cancer dataset files:
  - `cancer_patient_data_sets.xlsx`
  - `generalites_cancer.pdf`

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

3. **Ensure data files are present**:
   - `cancer_patient_data_sets.xlsx`
   - `generalites_cancer.pdf`

## ▶️ Running the Web Interface

```bash
python app.py
```

The server will start on `http://localhost:5000`

Open your browser and navigate to:
- **Chat Interface**: http://localhost:5000/
- **Health Check**: http://localhost:5000/api/health

## 📁 Project Structure

```
.
├── app.py                      # Flask backend server
├── OncoCheck.py               # Core RAG pipeline (unchanged)
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html            # Web interface HTML
├── static/
│   ├── style.css             # CSS styling
│   └── script.js             # Client-side JavaScript
└── README.md                 # This file
```

## 🎯 Usage

1. **Start the server** with `python app.py`
2. **Open the web interface** in your browser
3. **Type your question** in the input box
4. **Press `Ctrl+Enter`** or click **Send** to submit
5. **View responses** from OncoCheck in the chat area

### Example Questions

- "What are the main cancer risk factors?"
- "How is cancer risk assessed?"
- "What are the symptoms of high cancer risk?"
- "Tell me about risk factor correlations"
- "What lifestyle factors influence cancer risk?"

## 🔌 API Endpoints

### POST `/api/chat`
Send a query to the RAG pipeline.

**Request**:
```json
{
    "message": "What are the main cancer risk factors?"
}
```

**Response**:
```json
{
    "status": "success",
    "response": "Based on the cancer dataset, the main risk factors are...",
    "query": "What are the main cancer risk factors?"
}
```

### GET `/api/health`
Check the health status of the API.

**Response**:
```json
{
    "status": "healthy",
    "vectors_loaded": 250
}
```

## ⚙️ Configuration

### Backend (Flask)
- **Host**: 0.0.0.0 (accessible from anywhere)
- **Port**: 5000
- **Debug**: Enabled by default (disable in production)

### Frontend (JavaScript)
- **Auto-resize**: Input textarea expands automatically
- **Keyboard Shortcuts**: `Ctrl+Enter` to send messages
- **Scroll**: Auto-scrolls to latest messages

## ⚠️ Important Notes

1. **Medical Disclaimer**: OncoCheck is not a doctor and should not replace professional medical advice.
2. **API Key**: Keep your GROQ API key secure in `.env` file
3. **Data Privacy**: Patient data is processed securely via ChromaDB cloud
4. **Production Deployment**: Disable debug mode and use a production WSGI server (e.g., Gunicorn)

## 🚀 Deployment (Production)

For production deployment, use Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🐛 Troubleshooting

### Connection Error
- Ensure Flask server is running (`python app.py`)
- Check that port 5000 is not in use
- Try accessing http://localhost:5000 directly

### API Key Error
- Verify `.env` file exists in project root
- Ensure `GROQ_API_KEY` is set correctly
- Test key with: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GROQ_API_KEY'))"`

### No Data Loaded
- Check that Excel and PDF files exist in the project directory
- Verify file paths in `app.py`
- Check ChromaDB connection status at `/api/health`

## 📝 License

OncoCheck © 2024 - Cancer Risk Assessment Assistant

---

**Built with**: Flask • ChromaDB • Groq LLaMA • HTML/CSS/JavaScript
