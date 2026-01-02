import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# LOCAL EMBEDDINGS (No Rate Limits)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import os
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FDRE TVT AI System", 
    layout="wide", 
    page_icon="🇪🇹",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Coat_of_arms_of_Ethiopia.svg/1200px-Coat_of_arms_of_Ethiopia.svg.png", width=50)
    st.header("🇪🇹 TVT Control Panel")
    
    # API Key
    if 'GOOGLE_API_KEY' in st.secrets:
        api_key = st.secrets['GOOGLE_API_KEY']
        st.success("System Key Active 🟢")
    else:
        api_key = st.text_input("🔑 Enter Google API Key", type="password")

    st.divider()
    
    # Dashboard
    if st.session_state.processed_files:
        with st.expander(f"📚 Active Knowledge ({len(st.session_state.processed_files)} Files)", expanded=False):
            for f in st.session_state.processed_files:
                st.caption(f"✅ {f}")
    else:
        st.info("🧠 Brain is empty. Upload docs.")

    st.divider()

    # File Upload
    pdf_docs = st.file_uploader("📂 Add Documents", accept_multiple_files=True)
    
    if st.button("⚡ Add to Knowledge Base", type="primary", use_container_width=True):
        if not pdf_docs:
            st.warning("Please select files first.")
        else:
            with st.spinner("Processing locally (Unlimited Quota)..."):
                st.session_state.trigger_process = True
    
    # Reset
    if st.button("🗑️ Reset Brain", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processed_files = []
        st.session_state.vector_db = None
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        st.rerun()

    # Download
    if st.session_state.messages:
        chat_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button("💾 Download Chat", chat_str, "TVT_History.txt")

# --- 4. CORE FUNCTIONS ---

def get_pdf_text_with_metadata(pdf_docs):
    documents = []
    file_names = []
    for pdf in pdf_docs:
        if pdf.name not in st.session_state.processed_files:
            pdf_reader = PdfReader(pdf)
            file_names.append(pdf.name)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={"source": pdf.name, "page": i + 1}
                    )
                    documents.append(doc)
    return documents, file_names

def update_vector_store(documents):
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # USE LOCAL EMBEDDINGS (HuggingFace) - No Rate Limits!
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    if st.session_state.vector_db is not None:
        new_db = FAISS.from_documents(chunks, embedding=embeddings)
        st.session_state.vector_db.merge_from(new_db)
    else:
        st.session_state.vector_db = FAISS.from_documents(chunks, embedding=embeddings)
    
    st.session_state.vector_db.save_local("faiss_index")

def get_ai_response(user_question):
    if st.session_state.vector_db is None:
        return "System Error: Database lost. Please reset.", []

    # Search k=5 for context
    docs = st.session_state.vector_db.similarity_search(user_question, k=5)
    
    # PROMPT UPDATED FOR GEMINI-PRO
    prompt_template = """
    You are the Official AI Assistant for the FDRE TVT Institute.
    Answer the question based ONLY on the provided Context below.
    
    INSTRUCTIONS:
    1. If the user asks in Amharic, YOU MUST ANSWER IN AMHARIC. If English, use English.
    2. Format your answer with bullet points if listing requirements.
    3. If the answer is not in the context, say "Information not found in the documents."
    
    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    # --- FIX: Changed model to 'gemini-pro' which is stable ---
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    sources = []
    seen = set()
    for doc in docs:
        key = f"{doc.metadata.get('source')} (Page {doc.metadata.get('page')})"
        if key not in seen:
            sources.append(key)
            seen.add(key)
            
    return response["output_text"], sources

# --- 5. LOGIC HANDLER ---

if getattr(st.session_state, 'trigger_process', False):
    if not api_key:
        st.error("❌ API Key Missing")
    else:
        try:
            raw_docs, new_names = get_pdf_text_with_metadata(pdf_docs)
            if not raw_docs:
                st.warning("No new files to process.")
            else:
                update_vector_store(raw_docs)
                st.session_state.processed_files.extend(new_names)
                st.toast(f"Added {len(new_names)} new documents!", icon="📚")
        except Exception as e:
            st.error(f"Error: {e}")
    st.session_state.trigger_process = False

# --- 6. MAIN CHAT UI ---

st.title("🇪🇹 FDRE TVT Institute - Smart Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Verified Sources / ምንጮች"):
                for source in message["sources"]:
                    st.caption(f"📄 {source}")

if prompt := st.chat_input("Ask about Curriculum, OS, or Manuals..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_db is None:
        with st.chat_message("assistant"):
            st.error("🧠 The Knowledge Base is empty. Please upload documents in the Sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    full_response, sources = get_ai_response(prompt)
                    
                    placeholder = st.empty()
                    displayed_response = ""
                    for chunk in full_response.split():
                        displayed_response += chunk + " "
                        time.sleep(0.02)
                        placeholder.markdown(displayed_response + "▌")
                    placeholder.markdown(displayed_response)

                    if sources:
                        with st.expander("🔍 Verified Sources / ምንጮች"):
                            for source in sources:
                                st.caption(f"📄 {source}")

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
