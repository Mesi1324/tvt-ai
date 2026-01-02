import streamlit as st
from pypdf import PdfReader
# --- UPDATED IMPORT BELOW (Fixes the ModuleNotFoundError) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FDRE TVT AI Assistant", layout="wide")

# --- HEADER & CSS ---
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
    </style>
    """, unsafe_allow_html=True)

st.title("🇪🇹 FDRE TVT Institute Knowledge Base")
st.markdown("Upload Curriculum or Manuals and chat in **English or Amharic**.")

# --- SESSION STATE ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    if 'GOOGLE_API_KEY' in st.secrets:
        api_key = st.secrets['GOOGLE_API_KEY']
        st.success("API Key loaded ✅")
    else:
        api_key = st.text_input("Enter Google API Key", type="password")
    
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    process_btn = st.button("Submit & Process")
    
    if st.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()

# --- HELPER FUNCTIONS ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted: text += extracted
    return text

def get_text_chunks(text):
    # This now uses the updated import
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer strictly from the context. If the answer is not in the context, say "Information not found".
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def handle_user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("Please upload documents first!")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.session_state.history.append({"question": user_question, "answer": response["output_text"]})
    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- MAIN LOGIC ---
if process_btn and pdf_docs:
    if not api_key:
        st.error("API Key Missing")
    else:
        with st.spinner("Processing..."):
            try:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

# Display History
for chat in st.session_state.history:
    st.markdown(f'<div class="chat-message user">👤 {chat["question"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-message bot">🤖 {chat["answer"]}</div>', unsafe_allow_html=True)

user_question = st.text_input("Ask a question:", key="input")
if user_question and api_key:
    handle_user_input(user_question)
