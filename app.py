import streamlit as st
import base64
from config.config import UPLOAD_DIR
from utils.pdf_processor import process_pdf
from utils.vector_store import init_db, insert_chunks, semantic_search
from models.embeddings import get_embeddings
from models.llm import get_ai_response

# 1. Page Config x-x-x
st.set_page_config(page_title="Knowledge Engine", layout="wide", initial_sidebar_state="collapsed")

init_db()

# 2. CSS -------------x-x-x-x-
st.markdown("""
<style>
    /* Center the chat content and give it a max-width for readability */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Style the Chat Input to look more premium */
    .stChatInputContainer {
        padding-bottom: 30px;
    }

    /* Message bubbles */
    .stChatMessage {
        border: none !important;
        background-color: transparent !important;
    }

    /* Ensure PDF Iframe doesn't have weird borders */
    iframe {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if "messages" not in st.session_state: st.session_state.messages = []
if "doc_id" not in st.session_state: st.session_state.doc_id = None
if "pdf_bytes" not in st.session_state: st.session_state.pdf_bytes = None

# --- SIDEBAR: 
with st.sidebar:
    st.title("AI Settings")
    res_mode = st.radio("Response Mode", ["Concise", "Detailed"], index=1)
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Knowledge Base", type="pdf")
    if uploaded_file and st.session_state.doc_id is None:
        with st.spinner("Analyzing document..."):
            st.session_state.pdf_bytes = uploaded_file.getvalue()
            doc_id, raw_chunks = process_pdf(uploaded_file)
            valid_chunks = [c for c in raw_chunks if c["text"].strip()]
            valid_texts = [c["text"] for c in valid_chunks]
            
            if valid_texts:
                embs = get_embeddings(valid_texts)
                if len(embs) == len(valid_chunks):
                    for i, chunk in enumerate(valid_chunks):
                        chunk["emb"] = embs[i]
                    insert_chunks(doc_id, valid_chunks)
                    st.session_state.doc_id = doc_id
                    st.success("Document Indexed!")

    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.doc_id = None
        st.session_state.pdf_bytes = None
        st.rerun()

# --- MAIN LAYOUT ---
# We use [2, 1.2] to give the chat significantly more room than the PDF
has_pdf = st.session_state.doc_id is not None
col_chat, col_pdf = st.columns([2, 1.2] if has_pdf else [1, 0.001], gap="large")

with col_chat:
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # 1. Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for s in message["sources"]:
                        st.caption(f"Page {s['page']}: {s['text'][:200]}...")

    # 2. FIXED BOTTOM INPUT

    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() 

    # 3. AI Logic (Triggered if the last message is from user)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_prompt = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            context = ""
            sources = []
            
            # Retrieval
            if st.session_state.doc_id:
                with st.status("Searching PDF...", expanded=False):
                    q_emb = get_embeddings([user_prompt])[0]
                    sources = semantic_search(st.session_state.doc_id, q_emb)
                    context = "\n".join([s["text"] for s in sources])

            # Generation
            full_response = ""
            placeholder = st.empty()
            
            with st.status("Generating Answer...", expanded=True) as status:
                stream = get_ai_response(user_prompt, context, res_mode)
                for chunk in stream:
                    if chunk.candidates[0].content.parts:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'text') and part.text:
                                full_response += part.text
                                placeholder.markdown(full_response + "▌")
                status.update(label="Complete", state="complete", expanded=False)
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response, 
                "sources": sources
            })
            st.rerun() 

    st.markdown('</div>', unsafe_allow_html=True)

#  PDF PANEL ---
if has_pdf:
    with col_pdf:
        st.markdown("### 📄 Document Reference")
        
        base64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="850vh" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)