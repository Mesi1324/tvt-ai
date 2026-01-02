import streamlit as st
from pypdf import PdfReader  # <--- UPDATED: Uses the modern library
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FDRE TVT AI Assistant", layout="wide")

# --- CUSTOM CSS FOR CHAT UI ---
st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.bot {
        background-color: #f0f0f0;
    }
    .stButton>button {
        width: 100%; background-color: #4CAF50; color: white; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🇪🇹 FDRE TVT Institute Knowledge Base")
st.markdown("Upload Curriculum or Manuals and chat in **English or Amharic**.")

# --- INITIALIZE SESSION STATE (For Chat History) ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Secure API Key Handling
    if 'GOOGLE_API_KEY' in st.secrets:
        api_key = st.secrets['GOOGLE_API_KEY']
        st.success("API Key loaded from System ✅")
    else:
        api_key = st.text_input("Enter Google API Key", type="password")
        st.caption("Get Key from Google AI Studio")

    st.divider()
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    process_btn = st.button("Submit & Process")

    # Clear Chat Button
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()

# --- HELPER FUNCTIONS ---

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Multilingual model for English/Amharic
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an expert technical assistant for the FDRE TVT Institute.
    Answer the question strictly based on the provided context.
    If the answer is not in the context, state "Information not found in the provided documents."
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def handle_user_input(user_question):
    # Check if Vector DB exists
    if not os.path.exists("faiss_index"):
        st.error("⚠️ Please upload and process documents first! / እባክዎ መጀመሪያ ሰነዶችን ይጫኑ")
        return

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        with st.spinner("Thinking... / በማሰብ ላይ..."):
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            answer = response["output_text"]
            
            # Save to history
            st.session_state.history.append({"question": user_question, "answer": answer})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Check your API Key or Internet Connection.")

# --- MAIN LOGIC ---

# 1. Process Documents
if process_btn and pdf_docs:
    if not api_key:
        st.error("Please provide an API Key.")
    else:
        with st.spinner("Processing PDF Documents..."):
            try:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Documents Processed Successfully! ✅")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

# 2. Display Chat History
for chat in st.session_state.history:
    st.markdown(f'<div class="chat-message user">👤 <b>User:</b> {chat["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-message bot">🤖 <b>AI:</b> {chat["answer"]}</div>', unsafe_allow_html=True)

# 3. Chat Input
user_question = st.text_input("Ask a question:", key="input")

if user_question:
    if not api_key:
        st.warning("Please enter your API Key in the sidebar.")
    else:
        handle_user_input(user_question)
