# PDF ICH Expert Q&A

A reliable, production-grade Q&A application for querying a fixed set of ~60 PDF documents using RAG (Retrieval-Augmented Generation).

## Tech Stack
- **Frontend**: Streamlit
- **Vector Store**: FAISS (CPU)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (Local)
- **LLM**: OpenRouter (Llama 3.3 70b Free)
- **Framework**: LangChain

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- An OpenRouter API Key (Get one at [openrouter.ai](https://openrouter.ai/))

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory (or copy `.env.example`) and add your API key:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Add Documents
Place all your PDF documents in the `pdfs/` directory.

### 5. Run the Application
```bash
python -m streamlit run app.py
```

## How it Works
1. **Indexing**: On the first run, the app loads all PDFs from the `pdfs/` folder, splits them into chunks, and generates embeddings using a local model.
2. **Persistence**: The FAISS index is saved to the `vectorstore/` folder. Subsequent loads take only seconds.
3. **Retrieval**: When a question is asked, the system finds the 5 most relevant chunks from the documents.
4. **Grounded Generation**: The OpenRouter LLM generates an answer using *only* the provided context. If no answer is found, it uses the mandatory rejection phrases.

## Deployment to Streamlit Community Cloud

1. **Push to GitHub**: Push this entire folder (including `pdfs/` and `vectorstore/`) to a private/public GitHub repository.
2. **Deploy**: Click the **Deploy** button in the Streamlit app or go to [share.streamlit.io](https://share.streamlit.io).
3. **Secrets**: In the Streamlit dashboard, go to **Settings > Secrets** and add:
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_api_key_here"
   ```

### Persistent Database
By committing the `vectorstore/` folder to GitHub, your friends will **not** have to wait for the database to reload. The app will load the pre-built index locally from the repository, making it near-instant for every user.
