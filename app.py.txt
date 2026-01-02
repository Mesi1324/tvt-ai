import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="FDRE TVT AI", layout="wide")

st.markdown("""
    <style>
    .stButton>button {background-color: #4CAF50; color: white; width: 100%;}
    </style>
    """, unsafe_allow_html=True)

st.title("🇪🇹 FDRE TVT Institute Knowledge Base")
st.markdown("Upload documents and ask in **English or Amharic**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    if 'GOOGLE_API_KEY' in st.secrets:
        api_key = st.secrets['GOOGLE_API_KEY']
        st.success("API Key System Loaded ✅")
    else:
        api_key = st.text_input("Enter Google API Key", type="password")

    pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True)
    process_btn = st.button("Submit & Process")

# --- LOGIC ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text(): text += page.extract_text()
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_chain():
    prompt_template = """
    Answer strictly from the context. If answer is missing, say "Information not found".
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

if process_btn and pdf_docs:
    if not api_key: st.error("API Key Missing!")
    else:
        with st.spinner("Processing..."):
            raw = get_pdf_text(pdf_docs)
            chunks = get_chunks(raw)
            get_vector_store(chunks)
            st.success("Done!")

q = st.text_input("Ask a question:")
if q and api_key:
    if os.path.exists("faiss_index"):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(q)
        chain = get_chain()
        res = chain({"input_documents": docs, "question": q}, return_only_outputs=True)
        st.write(res["output_text"])
    else:
        st.warning("Please upload documents first.")

