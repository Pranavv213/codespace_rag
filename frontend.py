import streamlit as st
import requests
import json
import time
from typing import Optional

# Configure the page
st.set_page_config(
    page_title="Codespace RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better UI
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #dcf8c6;
        align-self: flex-end;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f1f0f0;
        align-self: flex-start;
        margin-right: 20%;
    }
    .message-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .stButton > button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None
if "current_index" not in st.session_state:
    st.session_state.current_index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Helper functions
def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_codebase(file, index_name):
    """Upload codebase to the API"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/zip")}
        data = {"index_name": index_name}
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=300  # 5 minutes timeout for large files
        )
        return response.json(), response.status_code
    except requests.exceptions.Timeout:
        return {"message": "Upload timed out. The file might be too large."}, 408
    except Exception as e:
        return {"message": f"Error: {str(e)}"}, 500

def send_chat_query(query, index_name, k=5):
    """Send a chat query to the API"""
    try:
        payload = {
            "query": query,
            "index_name": index_name,
            "k": k
        }
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=60
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": f"Error: {str(e)}"}, 500

def get_indices():
    """Get list of available indices"""
    try:
        response = requests.get(f"{API_BASE_URL}/indices", timeout=10)
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def delete_index(index_name):
    """Delete an index"""
    try:
        response = requests.delete(f"{API_BASE_URL}/index/{index_name}", timeout=10)
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

# Main UI
st.title("📚 Codebase RAG Assistant")
st.markdown("Upload your codebase and chat with it using AI!")

# Check API health
if not check_api_health():
    st.error("⚠️ Cannot connect to the API. Please make sure the FastAPI server is running on http://localhost:8000")
    st.info("Start the server with: python your_fastapi_file.py")
    st.stop()

# Sidebar - Upload Section
with st.sidebar:
    st.header("📤 Upload Codebase")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a zip file",
        type=['zip'],
        help="Upload a zip file containing your codebase (Python, JavaScript, TypeScript files)"
    )
    
    # Index name input
    index_name = st.text_input(
        "Index Name",
        placeholder="e.g., my-project-backend",
        help="Give a unique name for this codebase index"
    )
    
    # Upload button
    if st.button("🚀 Upload and Index", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Please select a zip file first.")
        elif not index_name.strip():
            st.error("Please enter an index name.")
        else:
            with st.spinner(f"Uploading and indexing '{uploaded_file.name}'... This may take a few moments."):
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Upload
                status_text.text("Step 1/3: Uploading file...")
                progress_bar.progress(33)
                
                result, status_code = upload_codebase(uploaded_file, index_name.strip())
                
                # Step 2: Processing
                status_text.text("Step 2/3: Processing and embedding...")
                progress_bar.progress(66)
                
                # Step 3: Complete
                status_text.text("Step 3/3: Finalizing...")
                progress_bar.progress(100)
                
                if status_code == 200:
                    if result.get("status") == "success":
                        st.session_state.upload_status = "success"
                        st.session_state.current_index = index_name.strip()
                        st.success(f"✅ Successfully embedded and created index '{index_name}'!")
                        st.info(f"📊 Uploaded {result.get('chunks_uploaded', 0)} chunks from {result.get('files_processed', 0)} files.")
                        
                        # Clear progress
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Refresh the indices list
                        st.rerun()
                    elif result.get("status") == "skipped":
                        st.warning(f"⚠️ Index '{index_name}' already exists. Please use a different name.")
                    else:
                        st.error(f"❌ Upload failed: {result.get('message', 'Unknown error')}")
                else:
                    st.error(f"❌ Upload failed with status {status_code}: {result.get('message', 'Unknown error')}")
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
    
    # Divider
    st.divider()
    
    # Index Management
    st.header("📋 Index Management")
    
    # Refresh indices button
    if st.button("🔄 Refresh Indices", use_container_width=True):
        with st.spinner("Refreshing..."):
            st.rerun()
    
    # Display available indices
    indices_data, status_code = get_indices()
    if status_code == 200:
        indices = indices_data.get("indices", [])
        if indices:
            st.success(f"✅ {len(indices)} indices available")
            for idx in indices:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📁 {idx}")
                with col2:
                    if st.button("🗑️", key=f"delete_{idx}", help=f"Delete index {idx}"):
                        if st.session_state.get(f"confirm_delete_{idx}", False):
                            with st.spinner(f"Deleting index '{idx}'..."):
                                result, status = delete_index(idx)
                                if status == 200:
                                    st.success(f"Deleted '{idx}'")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete: {result.get('message', 'Unknown error')}")
                        else:
                            st.session_state[f"confirm_delete_{idx}"] = True
                            st.warning(f"Click again to confirm deleting '{idx}'")
                            st.rerun()
        else:
            st.info("No indices available. Upload a codebase to get started.")
    else:
        st.error("Failed to fetch indices")
    
    # Current index info
    if st.session_state.current_index:
        st.divider()
        st.info(f"📌 Current Index: **{st.session_state.current_index}**")

# Main Chat Area
st.header("💬 Chat with your Codebase")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <b>You:</b> {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message assistant-message">
                    <b>Assistant:</b> {message["content"]}
                </div>
            """, unsafe_allow_html=True)

# Chat input
if st.session_state.current_index:
    # Show current index
    st.info(f"💡 Currently chatting with index: **{st.session_state.current_index}**")
    
    # Chat input row
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_query = st.text_input(
            "Ask a question about your codebase",
            placeholder="e.g., What authentication methods are used?",
            key="user_query_input"
        )
    
    with col2:
        submit_button = st.button("Send 📤", type="primary", use_container_width=True)
    
    # Handle submission
    if submit_button and user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Get response from API
        with st.spinner("Thinking..."):
            result, status_code = send_chat_query(
                user_query,
                st.session_state.current_index,
                k=5
            )
            
            if status_code == 200:
                answer = result.get("answer", "No answer received.")
                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                error_msg = f"❌ Error: {result.get('detail', 'Unknown error')}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })
        
        # Rerun to update the chat display
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
else:
    st.warning("⚠️ Please upload a codebase first to start chatting. Use the sidebar to upload and create an index.")

# Footer
st.divider()
st.caption("🔒 All sensitive data is sanitized and masked. Powered by Pinecone, Gemini, and LangChain.")

# Optional: Add a button to switch to a different index
if st.session_state.current_index and st.sidebar.button("🔄 Switch Index", use_container_width=True):
    st.session_state.current_index = None
    st.session_state.chat_history = []
    st.rerun()