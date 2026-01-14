import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
import time

# Load environment variables
load_dotenv()

# Constants
PDF_DIRECTORY = "pdfs/"
VECTORSTORE_PATH = "vectorstore/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Free reliable model on OpenRouter
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

def get_openrouter_client():
    api_key = None
    
    # 1. Check Streamlit Secrets (Priority for Cloud)
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            api_key = st.secrets["OPENROUTER_API_KEY"]
    except:
        pass
        
    # 2. Check Environment (Priority for Local)
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")
        
    if not api_key:
        st.sidebar.error("🔑 API Key not found in Secrets!")
        return None
        
    key = api_key.strip()
    
    # Visual confirmation in sidebar (safe)
    st.sidebar.success(f"🔑 Key detected (starts with {key[:10]}...)")
    
    if not key.startswith("sk-or-v1-"):
        st.sidebar.warning("⚠️ Warning: Key should typically start with 'sk-or-v1-'. Please check for typos or extra spaces.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
    )

def build_vector_store():
    """Builds and saves the FAISS vector store from PDFs."""
    if not os.path.exists(PDF_DIRECTORY):
        os.makedirs(PDF_DIRECTORY)
        st.warning(f"No '{PDF_DIRECTORY}' folder found. Created it. Please add your PDFs and restart.")
        st.stop()

    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith(".pdf")]
    if not pdf_files:
        st.error(f"No PDF files found in '{PDF_DIRECTORY}'. Please add some PDFs to continue.")
        st.stop()

    st.info(f"Found {len(pdf_files)} PDFs. Building index... This may take a few minutes for the first run.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Load PDFs
    status_text.text("Loading PDFs...")
    loader = DirectoryLoader(PDF_DIRECTORY, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    progress_bar.progress(30)

    # Split text
    status_text.text("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    progress_bar.progress(60)

    # Create embeddings and vector store
    status_text.text("Generating embeddings and building FAISS index (CPU)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk
    vector_store.save_local(VECTORSTORE_PATH)
    progress_bar.progress(100)
    status_text.text("Index built and saved successfully!")
    time.sleep(2)
    status_text.empty()
    progress_bar.empty()
    
    return vector_store

def load_vector_store():
    """Loads the FAISS vector store from disk or builds it if it doesn't exist."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return build_vector_store()

def generate_answer(query, vector_store, client):
    """Retrieves context and generates a grounded answer using OpenRouter."""
    # Retrieve relevant chunks
    docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})\nContent: {doc.page_content}" for doc in docs])

    system_prompt = """
You are a strictly honest and professional ICH expert assistant.
You provide direct, factual information based on the expert knowledge provided to you.

Rules — you MUST obey:
1. Answer ONLY using the context below.
2. DO NOT mention "PDFs", "documents", or "the provided context" in your answer. Act as if the knowledge is your own.
3. If the question is not answerable from the provided context → reply ONLY with:
   - "I'm sorry, I don't have that information available."
   - OR "This information is not present in my expert knowledge base."
4. Be concise, direct, and factual.
5. Do not use your general knowledge or make assumptions.

Context:
{context}

Question:
{input}

Answer:
"""
    
    formatted_prompt = system_prompt.format(context=context, input=query)

    if not client:
        return "I'm sorry, I cannot answer questions without a valid API key. Please check the sidebar for details.", []

    try:
        completion = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": "You are a strictly honest and professional ICH expert assistant."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0,
            max_tokens=1024,
        )
        return completion.choices[0].message.content, docs
    except Exception as e:
        return f"Error connecting to expert knowledge base: {str(e)}", []

def main():
    st.set_page_config(page_title="ICH Expert", layout="wide", page_icon="🧬")
    
    st.title("🧬 ICH Expert")
    
    # Initialize session state for vector store and client
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store()
    
    if "client" not in st.session_state:
        st.session_state.client = get_openrouter_client()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                # Create a concise source citation
                source_list = []
                for doc in message["sources"]:
                    source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                    page_num = doc.metadata.get("page", "N/A")
                    source_list.append(f"{source_name} (p. {page_num})")
                
                # Deduplicate and join
                unique_sources = sorted(list(set(source_list)))
                st.caption(f"Sources: {', '.join(unique_sources)}")

    # React to user input
    if prompt := st.chat_input("How can I help you with ICH guidelines?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                answer, sources = generate_answer(prompt, st.session_state.vector_store, st.session_state.client)
                st.markdown(answer)
                
                if sources:
                    # Create a concise source citation
                    source_list = []
                    for doc in sources:
                        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                        page_num = doc.metadata.get("page", "N/A")
                        source_list.append(f"{source_name} (p. {page_num})")
                    
                    unique_sources = sorted(list(set(source_list)))
                    citation = f"Sources: {', '.join(unique_sources)}"
                    st.caption(citation)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })

    # Footer/Sidebar
    with st.sidebar:
        st.header("Actions")
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
            
        with st.expander("Advanced Settings"):
            if st.button("Refresh Knowledge Base"):
                st.session_state.vector_store = build_vector_store()
                st.rerun()

if __name__ == "__main__":
    main()
