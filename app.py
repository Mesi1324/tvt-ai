import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
# We use this to remember files across reruns so we don't lose data
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = [] # Track names of indexed files
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None # Store the FAISS DB in RAM

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
    
    # --- UPGRADE: Active Knowledge Dashboard ---
    # Shows the user what the AI currently knows
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
            with st.spinner("Processing & Merging..."):
                # Trigger the processing logic
                st.session_state.trigger_process = True
    
    # Reset
    if st.button("🗑️ Reset Brain (Clear All)", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processed_files = []
        st.session_state.vector_db = None
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index") # Delete local file
        st.rerun()

    # Download Chat
    if st.session_state.messages:
        chat_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button("💾 Download Chat", chat_str, "TVT_History.txt")

# --- 4. CORE FUNCTIONS ---

def get_pdf_text_with_metadata(pdf_docs):
    documents = []
    file_names = []
    for pdf in pdf_docs:
        # Avoid re-processing files we already have (Simple check)
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
    """
    UPGRADE: Merges new documents into existing DB instead of overwriting.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # If DB exists, add to it. If not, create new.
    if st.session_state.vector_db is not None:
        new_db = FAISS.from_documents(chunks, embedding=embeddings)
        st.session_state.vector_db.merge_from(new_db)
    else:
        st.session_state.vector_db = FAISS.from_documents(chunks, embedding=embeddings)
    
    # Save to local disk for redundancy
    st.session_state.vector_db.save_local("faiss_index")

def get_ai_response(user_question):
    # Use the session_state DB for speed
    if st.session_state.vector_db is None:
        return "System Error: Database lost. Please reset.", []

    # UPGRADE: Search k=8 chunks for deeper analysis
    docs = st.session_state.vector_db.similarity_search(user_question, k=8)
    
    # UPGRADE: Robust Prompt
    prompt_template = """
    You are the Official AI Assistant for the FDRE TVT Institute.
    
    INSTRUCTIONS:
    1. Language Rule: If the user asks in Amharic, YOU MUST ANSWER IN AMHARIC. If English, use English.
    2. Context Rule: Answer ONLY based on the provided documents.
    3. Formatting: Use clear headers and bullet points.
    4. Honesty: If the answer is not in the context, say "Information not found in the documents."
    
    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Source deduplication
    sources = []
    seen = set()
    for doc in docs:
        key = f"{doc.metadata.get('source')} (Page {doc.metadata.get('page')})"
        if key not in seen:
            sources.append(key)
            seen.add(key)
            
    return response["output_text"], sources

# --- 5. LOGIC HANDLER ---

# Check if processing was triggered
if getattr(st.session_state, 'trigger_process', False):
    if not api_key:
        st.error("❌ API Key Missing")
    else:
        try:
            # 1. Extract Text
            raw_docs, new_names = get_pdf_text_with_metadata(pdf_docs)
            
            if not raw_docs:
                st.warning("No new files to process (Duplicate or Empty).")
            else:
                # 2. Update Vector DB (Merge)
                update_vector_store(raw_docs)
                
                # 3. Update File List
                st.session_state.processed_files.extend(new_names)
                st.toast(f"Added {len(new_names)} new documents!", icon="📚")
                
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Reset trigger
    st.session_state.trigger_process = False


# --- 6. MAIN CHAT UI ---

st.title("🇪🇹 FDRE TVT Institute - Smart Assistant")

# Chat Container
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Verified Sources / ምንጮች"):
                for source in message["sources"]:
                    st.caption(f"📄 {source}")

# Input
if prompt := st.chat_input("Ask about Curriculum, OS, or Manuals..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check DB
    if st.session_state.vector_db is None:
        with st.chat_message("assistant"):
            st.error("🧠 The Knowledge Base is empty. Please upload documents in the Sidebar.")
    else:
        # AI Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    full_response, sources = get_ai_response(prompt)
                    
                    # Stream output
                    placeholder = st.empty()
                    displayed_response = ""
                    for chunk in full_response.split():
                        displayed_response += chunk + " "
                        time.sleep(0.02)
                        placeholder.markdown(displayed_response + "▌")
                    placeholder.markdown(displayed_response)

                    # Show sources
                    if sources:
                        with st.expander("🔍 Verified Sources / ምንጮች"):
                            for source in sources:
                                st.caption(f"📄 {source}")

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
