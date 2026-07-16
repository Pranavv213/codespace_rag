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
MAX_FILE_SIZE_MB = 1.5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # 1.5 MB in bytes

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
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.5rem;
        color: #856404;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        color: #0c5460;
    }
    .file-context {
        padding: 0.5rem 1rem;
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .file-context strong {
        color: #0066cc;
    }
    .index-selector {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .file-size-warning {
        padding: 0.5rem 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.25rem;
        color: #856404;
        font-size: 0.9rem;
        margin-top: 0.5rem;
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
if "current_file_context" not in st.session_state:
    st.session_state.current_file_context = None
if "available_indices" not in st.session_state:
    st.session_state.available_indices = []
if "use_existing_index" not in st.session_state:
    st.session_state.use_existing_index = False

# Helper functions
def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False
    
def get_file_size_mb(file):
    """Get file size in MB"""
    file.seek(0, 2)  # Seek to end
    size_bytes = file.tell()
    file.seek(0)  # Reset to beginning
    return size_bytes / (1024 * 1024)

def validate_file_size(file):
    """Validate file size against max limit"""
    size_mb = get_file_size_mb(file)
    return size_mb <= MAX_FILE_SIZE_MB, size_mb

def upload_codebase(file, index_name):
    """Upload codebase to the API with improved error handling"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/zip")}
        data = {"index_name": index_name}
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=300
        )
        
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = {"message": "Invalid response from server"}
            
        return response_data, response.status_code
        
    except requests.exceptions.Timeout:
        return {
            "message": "The upload is taking too long. Your codebase might be too large.",
            "user_friendly": True,
            "suggestion": "Try compressing your codebase further or splitting it into smaller parts."
        }, 408
        
    except requests.exceptions.ConnectionError:
        return {
            "message": "Cannot connect to the server. Please make sure the backend is running.",
            "user_friendly": True,
            "suggestion": "Check if the FastAPI server is running on http://localhost:8000"
        }, 503
        
    except requests.exceptions.RequestException as e:
        return {
            "message": f"Network error occurred: {str(e)}",
            "user_friendly": True,
            "suggestion": "Please check your network connection and try again."
        }, 500
        
    except Exception as e:
        return {
            "message": f"An unexpected error occurred: {str(e)}",
            "user_friendly": True,
            "suggestion": "Please try again. If the problem persists, contact support."
        }, 500

def send_chat_query(query, index_name, file_name=None, k=5):
    """Send a chat query to the API with improved error handling"""
    try:
        payload = {
            "query": query,
            "index_name": index_name,
            "k": k
        }
        
        if file_name:
            payload["file_name"] = file_name
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=60
        )
        
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = {"error": "Invalid response from server"}
            
        return response_data, response.status_code
        
    except requests.exceptions.Timeout:
        return {
            "error": "The request is taking too long to process.",
            "user_friendly": True,
            "suggestion": "Try asking a more specific question or reducing the number of results (k)."
        }, 408
        
    except requests.exceptions.ConnectionError:
        return {
            "error": "Cannot connect to the server. Please check if the backend is running.",
            "user_friendly": True,
            "suggestion": "Make sure the FastAPI server is running on http://localhost:8000"
        }, 503
        
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Network error: {str(e)}",
            "user_friendly": True,
            "suggestion": "Please check your network connection and try again."
        }, 500
        
    except Exception as e:
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "user_friendly": True,
            "suggestion": "Please try again later."
        }, 500

def get_indices():
    """Get list of available indices"""
    try:
        response = requests.get(f"{API_BASE_URL}/indices", timeout=10)
        try:
            return response.json(), response.status_code
        except json.JSONDecodeError:
            return {"error": "Invalid response from server"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

def delete_index(index_name):
    """Delete an index"""
    try:
        response = requests.delete(f"{API_BASE_URL}/index/{index_name}", timeout=10)
        try:
            return response.json(), response.status_code
        except json.JSONDecodeError:
            return {"error": "Invalid response from server"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

def get_files_in_index(index_name):
    """Get list of files in an index"""
    try:
        response = requests.get(f"{API_BASE_URL}/index/{index_name}/files", timeout=10)
        try:
            return response.json(), response.status_code
        except json.JSONDecodeError:
            return {"error": "Invalid response from server"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

def verify_index_exists(index_name):
    """Verify if an index exists"""
    indices_data, status_code = get_indices()
    if status_code == 200:
        indices = indices_data.get("indices", [])
        return index_name in indices
    return False

def display_error_message(error_data, default_message="An error occurred"):
    """Display user-friendly error messages"""
    if isinstance(error_data, dict):
        if "rate limit" in str(error_data).lower() or "too many requests" in str(error_data).lower():
            st.error("🚫 Rate Limit Exceeded")
            st.info("""
            **What happened:** You've made too many requests in a short period.
            
            **What to do:**
            - Wait a few minutes before trying again
            - If uploading, try a smaller file
            - If asking questions, try to be more specific
            """)
            return
            
        if "api key" in str(error_data).lower() or "authentication" in str(error_data).lower():
            st.error("🔑 API Key Error")
            st.info("""
            **What happened:** There's an issue with the API authentication.
            
            **What to do:**
            - Check if your API keys are properly configured
            - Verify that your API keys have the necessary permissions
            - Contact your system administrator
            """)
            return
            
        if "quota" in str(error_data).lower() or "limit exceeded" in str(error_data).lower():
            st.error("📊 Quota Exceeded")
            st.info("""
            **What happened:** You've reached the usage limit for this service.
            
            **What to do:**
            - Wait until the quota resets
            - Contact your administrator for increased limits
            - Consider using a different service tier
            """)
            return
            
        if "not found" in str(error_data).lower() or "does not exist" in str(error_data).lower():
            st.error("📁 Index Not Found")
            st.info("""
            **What happened:** The requested codebase index doesn't exist.
            
            **What to do:**
            - Check if you've uploaded the codebase successfully
            - Refresh the indices list in the sidebar
            - Upload the codebase again with a different name
            """)
            return
        
        if "invalid" in str(error_data).lower() or "corrupt" in str(error_data).lower():
            st.error("📄 Invalid File Format")
            st.info("""
            **What happened:** The uploaded file appears to be invalid or corrupted.
            
            **What to do:**
            - Make sure you're uploading a valid ZIP file
            - Try re-compressing your codebase
            - Check if the file is not empty or corrupted
            """)
            return
            
        if "size" in str(error_data).lower() or "large" in str(error_data).lower():
            st.error("📦 File Too Large")
            st.info(f"""
            **What happened:** The uploaded file exceeds the size limit of {MAX_FILE_SIZE_MB} MB.
            
            **What to do:**
            - Compress your codebase to reduce file size
            - Remove unnecessary files (node_modules, __pycache__, etc.) before compressing
            - Split your codebase into smaller parts
            - Only include essential source code files
            """)
            return
    
    if isinstance(error_data, dict):
        message = error_data.get("message", default_message)
        suggestion = error_data.get("suggestion", "")
        
        if error_data.get("user_friendly", False):
            st.error(f"❌ {message}")
            if suggestion:
                st.info(f"💡 **Suggestion:** {suggestion}")
        else:
            st.error(f"❌ {message}")
    else:
        st.error(f"❌ {str(error_data)}")

def display_file_size_warning(file_size_mb):
    """Display file size warning with visual indicator"""
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.markdown(f"""
            <div class="file-size-warning">
                ⚠️ <strong>File too large!</strong> Your file is {file_size_mb:.2f} MB. 
                Maximum allowed size is {MAX_FILE_SIZE_MB} MB.
                <br>Please reduce the file size and try again.
            </div>
        """, unsafe_allow_html=True)
    elif file_size_mb > MAX_FILE_SIZE_MB * 0.8:  # Warning when file is > 80% of limit
        st.markdown(f"""
            <div class="file-size-warning" style="background-color: #fff3cd; border-color: #ffeeba; color: #856404;">
                ⚠️ <strong>Large file:</strong> Your file is {file_size_mb:.2f} MB. 
                Maximum allowed size is {MAX_FILE_SIZE_MB} MB.
                <br>Consider compressing it further if you encounter issues.
            </div>
        """, unsafe_allow_html=True)

# Main UI
st.title("📚 Codebase RAG Assistant")
st.markdown("Upload your codebase and chat with it using AI!")

# Display file size limit prominently
st.info(f"📦 **File Size Limit:** Maximum upload size is **{MAX_FILE_SIZE_MB} MB** per ZIP file. Please compress your codebase appropriately.")

# Check API health
if not check_api_health():
    st.error("⚠️ Cannot connect to the API. Please make sure the FastAPI server is running on http://localhost:8000")
    st.info("💡 **How to fix:** Start the server with: `python your_fastapi_file.py`")
    st.stop()

# Sidebar - Index Selection & Upload Section
with st.sidebar:
    st.header("🔍 Select or Create Index")
    
    # Fetch available indices
    indices_data, status_code = get_indices()
    if status_code == 200:
        st.session_state.available_indices = indices_data.get("indices", [])
    
    # Option to use existing index or create new
    use_existing = st.radio(
        "Choose an option:",
        ["📂 Use Existing Index", "📤 Create New Index"],
        index=0 if st.session_state.current_index else 1,
        key="index_option"
    )
    
    if "📂 Use Existing Index" in use_existing:
        st.session_state.use_existing_index = True
        
        if st.session_state.available_indices:
            # Dropdown to select existing index
            selected_index = st.selectbox(
                "Select an existing index:",
                options=st.session_state.available_indices,
                help="Choose an index you've previously uploaded"
            )
            
            # Button to load the selected index
            if st.button("📂 Load Selected Index", type="primary", use_container_width=True):
                if selected_index:
                    st.session_state.current_index = selected_index
                    st.session_state.chat_history = []
                    st.session_state.current_file_context = None
                    st.success(f"✅ Loaded index: {selected_index}")
                    st.rerun()
                else:
                    st.warning("Please select an index from the list.")
            
            # Quick select buttons for recent indices
            if len(st.session_state.available_indices) > 0:
                st.markdown("---")
                st.markdown("**Quick Load:**")
                for idx in st.session_state.available_indices[:5]:  # Show first 5
                    if st.button(f"📁 {idx}", key=f"quick_{idx}", use_container_width=True):
                        st.session_state.current_index = idx
                        st.session_state.chat_history = []
                        st.session_state.current_file_context = None
                        st.success(f"✅ Loaded index: {idx}")
                        st.rerun()
        else:
            st.info("📭 No indices available. Create a new one by uploading a codebase.")
            st.session_state.use_existing_index = False
    
    else:  # Create New Index
        st.session_state.use_existing_index = False
        st.markdown("---")
        st.subheader("📤 Upload New Codebase")
        
        # Show file size limit in upload section
        st.caption(f"📦 Max file size: {MAX_FILE_SIZE_MB} MB")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a zip file",
            type=['zip'],
            help=f"Upload a zip file containing your codebase. Maximum size: {MAX_FILE_SIZE_MB} MB"
        )
        
        # Check file size when uploaded
        if uploaded_file is not None:
            file_size_mb = get_file_size_mb(uploaded_file)
            is_valid, _ = validate_file_size(uploaded_file)
            
            # Display file info
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"📄 **File:** {uploaded_file.name}")
            with col2:
                st.write(f"📊 **Size:** {file_size_mb:.2f} MB")
            
            # Show warning if file is too large
            if not is_valid:
                display_file_size_warning(file_size_mb)
                st.error(f"❌ This file exceeds the {MAX_FILE_SIZE_MB} MB limit. Please compress it further.")
            elif file_size_mb > MAX_FILE_SIZE_MB * 0.8:
                display_file_size_warning(file_size_mb)
        
        # Index name input
        index_name = st.text_input(
            "Index Name",
            placeholder="e.g., my-project-backend",
            help="Give a unique name for this codebase index"
        )
        
        # Upload button with validation
        upload_disabled = uploaded_file is None or not index_name.strip()
        if uploaded_file is not None:
            is_valid, _ = validate_file_size(uploaded_file)
            if not is_valid:
                upload_disabled = True
        
        if st.button("🚀 Upload and Index", type="primary", use_container_width=True, disabled=upload_disabled):
            if uploaded_file is None:
                st.error("📁 Please select a zip file first.")
            elif not index_name.strip():
                st.error("✏️ Please enter an index name.")
            else:
                # Validate file size again before upload
                is_valid, file_size_mb = validate_file_size(uploaded_file)
                if not is_valid:
                    st.error(f"❌ File size ({file_size_mb:.2f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit.")
                    st.info("💡 Please compress your codebase to reduce file size.")
                else:
                    # Check if index already exists
                    if verify_index_exists(index_name.strip()):
                        st.warning(f"⚠️ Index '{index_name}' already exists. Please use a different name or load the existing one.")
                        st.info("💡 **Tip:** Switch to 'Use Existing Index' mode to load it.")
                    else:
                        with st.spinner(f"Uploading and indexing '{uploaded_file.name}'... This may take a few moments."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Step 1/3: Uploading file...")
                            progress_bar.progress(33)
                            
                            result, status_code = upload_codebase(uploaded_file, index_name.strip())
                            
                            status_text.text("Step 2/3: Processing and embedding...")
                            progress_bar.progress(66)
                            
                            status_text.text("Step 3/3: Finalizing...")
                            progress_bar.progress(100)
                            
                            if status_code == 200:
                                if result.get("status") == "success":
                                    st.session_state.upload_status = "success"
                                    st.session_state.current_index = index_name.strip()
                                    st.session_state.current_file_context = None
                                    st.success(f"✅ Successfully uploaded and indexed '{index_name}'!")
                                    st.info(f"📊 Uploaded {result.get('chunks_uploaded', 0)} chunks from {result.get('files_processed', 0)} files.")
                                    st.info(f"📦 File size: {file_size_mb:.2f} MB")
                                    
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.rerun()
                                elif result.get("status") == "skipped":
                                    st.warning(f"⚠️ Index '{index_name}' already exists. Please use a different name.")
                                    st.info("💡 **Tip:** Try using a different name or delete the existing index first.")
                                else:
                                    display_error_message(
                                        result.get("message", "Unknown error occurred"),
                                        "Upload failed"
                                    )
                            elif status_code == 408:
                                st.error("⏱️ Upload Timed Out")
                                st.info("""
                                **What happened:** The upload took too long to complete.
                                
                                **What to do:**
                                - Try uploading a smaller codebase
                                - Check your internet connection speed
                                - Compress your codebase further
                                """)
                            elif status_code == 503:
                                st.error("🔌 Server Unavailable")
                                st.info("""
                                **What happened:** The backend server is not accessible.
                                
                                **What to do:**
                                - Make sure the FastAPI server is running
                                - Check if the server is on the correct port (8000)
                                - Verify your firewall settings
                                """)
                            elif status_code == 413:
                                st.error("📦 File Too Large")
                                st.info(f"""
                                **What happened:** The uploaded file exceeds the server's size limit.
                                
                                **What to do:**
                                - Remove unnecessary files from your codebase
                                - Use a more aggressive compression
                                - Split your codebase into multiple parts
                                - Current limit: {MAX_FILE_SIZE_MB} MB
                                """)
                            elif status_code == 429:
                                st.error("🚫 Rate Limit Exceeded")
                                st.info("""
                                **What happened:** You've made too many upload requests.
                                
                                **What to do:**
                                - Wait a few minutes before trying again
                                - Reduce the frequency of your uploads
                                """)
                            else:
                                error_msg = result.get("message", f"Unknown error (Status: {status_code})")
                                st.error(f"❌ Upload failed: {error_msg}")
                                st.info("💡 Check the server logs for more details about this error.")
                            
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
    if st.session_state.available_indices:
        st.success(f"✅ {len(st.session_state.available_indices)} indices available")
        for idx in st.session_state.available_indices:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Highlight current index
                if idx == st.session_state.current_index:
                    st.markdown(f"📁 **{idx}** (active)")
                else:
                    st.text(f"📁 {idx}")
            with col2:
                if st.button("🗑️", key=f"delete_{idx}", help=f"Delete index {idx}"):
                    if st.session_state.get(f"confirm_delete_{idx}", False):
                        with st.spinner(f"Deleting index '{idx}'..."):
                            result, status = delete_index(idx)
                            if status == 200:
                                st.success(f"✅ Deleted '{idx}' successfully")
                                if st.session_state.current_index == idx:
                                    st.session_state.current_index = None
                                    st.session_state.chat_history = []
                                    st.session_state.current_file_context = None
                                st.rerun()
                            else:
                                display_error_message(
                                    result.get("error", "Unknown error"),
                                    "Failed to delete index"
                                )
                    else:
                        st.session_state[f"confirm_delete_{idx}"] = True
                        st.warning(f"⚠️ Click again to confirm deleting '{idx}'")
                        st.rerun()
    else:
        st.info("📭 No indices available.")
    
    # Current index info
    if st.session_state.current_index:
        st.divider()
        st.info(f"📌 Current Index: **{st.session_state.current_index}**")
        
        # Show files in the index
        if st.button("📂 View Files in Index", use_container_width=True):
            with st.spinner("Fetching files..."):
                files_data, status = get_files_in_index(st.session_state.current_index)
                if status == 200:
                    files = files_data.get("files", [])
                    if files:
                        st.success(f"📄 {len(files)} files found")
                        file_list = "\n".join([f"• {f}" for f in files[:20]])
                        if len(files) > 20:
                            file_list += f"\n... and {len(files) - 20} more files"
                        st.code(file_list, language="text")
                    else:
                        st.info("No files found in this index")
                else:
                    st.error("Failed to fetch files")
        
        # Switch index button
        if st.button("🔄 Switch Index", use_container_width=True):
            st.session_state.current_index = None
            st.session_state.chat_history = []
            st.session_state.current_file_context = None
            st.rerun()

# Main Chat Area
st.header("💬 Chat with your Codebase")

# Show current context
if st.session_state.current_index:
    # Display current index and file context
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"💡 Currently chatting with index: **{st.session_state.current_index}**")
        if st.session_state.current_file_context:
            st.markdown(f"""
                <div class="file-context">
                    📄 <strong>Current file context:</strong> {st.session_state.current_file_context}
                    <br><small>Questions will be focused on this specific file</small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="file-context" style="border-left-color: #ff9900;">
                    📁 <strong>Searching across all files</strong>
                    <br><small>Specify a file name below to focus on a specific file</small>
                </div>
            """, unsafe_allow_html=True)

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                file_context = message.get("file_context", "")
                if file_context:
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.3rem;">
                                📄 File: {file_context}
                            </div>
                            <b>You:</b> {message["content"]}
                        </div>
                    """, unsafe_allow_html=True)
                else:
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
    with st.form(key="chat_form", clear_on_submit=True):
        # File context input
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**📄 File Context**")
            st.caption("Optional")
        
        with col2:
            file_name = st.text_input(
                "File name to focus on",
                placeholder="e.g., main.py, utils/helper.js",
                value=st.session_state.current_file_context if st.session_state.current_file_context else "",
                key="file_context_input",
                help="Specify a file name to focus your question on that specific file. Leave empty to search across all files."
            )
        
        # User query input
        user_query = st.text_input(
            "Ask a question about your codebase",
           
            key="user_query_input"
        )
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            submit_button = st.form_submit_button("Send 📤", type="primary", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
        
        # Handle form submission
        if submit_button and user_query:
            file_context = file_name.strip() if file_name else None
            if file_context:
                st.session_state.current_file_context = file_context
            
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query,
                "file_context": file_context
            })
            
            with st.spinner("🧠 Thinking..."):
                result, status_code = send_chat_query(
                    user_query,
                    st.session_state.current_index,
                    file_name=file_context if file_context else None,
                    k=5
                )
                
                if status_code == 200:
                    answer = result.get("answer", "No answer received.")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                elif status_code == 408:
                    error_msg = "⏱️ The request timed out. Try asking a more specific question."
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                elif status_code == 429:
                    error_msg = "🚫 Rate limit exceeded. Please wait a moment before asking another question."
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                else:
                    error_msg = result.get("error", f"⚠️ Error {status_code}: Could not process your question.")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })
            
            st.rerun()
        
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.current_file_context = None
            st.rerun()
    
    # Tips section
    with st.expander("💡 Tips for better results"):
        st.markdown("""
        **📁 Using File Context:**
        - Specify a file name to get answers focused on that specific file
        - Examples: `main.py`, `src/utils/helper.js`, `models/user.py`
        - Leave empty to search across your entire codebase
        
        **❓ Asking Questions:**
        - Be specific about what you want to know
        - Include class names, function names, or file paths if relevant
        - Example: "What does the `authenticate()` function in auth.py do?"
        
        **📊 Understanding Responses:**
        - The assistant will show which files it used to generate the answer
        - Responses include context from your codebase
        - Ask follow-up questions to dig deeper into specific areas
        
        **📦 File Size Tips:**
        - Keep your ZIP file under {MAX_FILE_SIZE_MB} MB
        - Remove unnecessary files (node_modules, __pycache__, .git, etc.)
        - Only include source code files relevant to your project
        - Use tools like `zip -r --exclude=*.pyc --exclude=__pycache__` to exclude unnecessary files
        """)
    
else:
    st.warning("⚠️ Please select or upload a codebase first to start chatting.")
    st.info("💡 Use the sidebar to either:")
    st.markdown(f"""
    1. **Use an existing index** - Select from the list of available indices
    2. **Create a new index** - Upload a new codebase zip file (max {MAX_FILE_SIZE_MB} MB)
    """)

# Footer
st.divider()
st.caption(f"🔒 All sensitive data is sanitized and masked. Powered by Pinecone, Gemini, and LangChain. | Max file size: {MAX_FILE_SIZE_MB} MB")

# Tips for reducing file size in the footer
