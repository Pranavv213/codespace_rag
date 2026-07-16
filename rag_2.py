import os
import zipfile
import ast
import sqlite3
import time
import re
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec

# Core & Integrations
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langsmith import traceable
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load environment keys from .env file automatically
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Codebase RAG API", description="API for highly optimized codebase indexing and querying")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend origin (HTML file, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PATCH, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all HTTP headers
)


# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    index_name: str
    k: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    query: str
    index_name: str

class UploadResponse(BaseModel):
    message: str
    index_name: str
    chunks_uploaded: int
    files_processed: int
    status: str

# 1. SQLite Setup for Context Storage
# ------------------------------------------------------------------
DB_PATH = "codebase_context.db"

def init_db():
    """Initializes the SQLite database table for file storage."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_content (
                index_name TEXT,
                filename TEXT,
                content TEXT,
                PRIMARY KEY (index_name, filename)
            )
        """)
init_db()

def save_to_db(index_name: str, filename: str, content: str):
    """Saves or updates a file's content in the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO file_content (index_name, filename, content) VALUES (?, ?, ?)",
            (index_name, filename, content)
        )

def get_from_db(index_name: str, filename: str) -> Optional[str]:
    """Retrieves a file's content from the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT content FROM file_content WHERE index_name=? AND filename=?",
            (index_name, filename)
        )
        row = cursor.fetchone()
        return row[0] if row else None

# 2. Setup Pinecone and AI Components
# ------------------------------------------------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
dimension = 1024  # Required dimension size for multilingual-e5-large

embeddings = PineconeEmbeddings(model="multilingual-e5-large")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)

# Global dictionary to store instantiated PineconeVectorStores
vectorstores = {}

@traceable
def sanitize_text(text: str) -> str:
    """Masks potential API keys, tokens, or common PII patterns."""
    patterns = [
        (r"(?i)(api[_-]?key|token|secret|password|passwd|pwd)\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{16,64}['\"]?", r"\1: [MASKED]"),
        (r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", r"\1 [MASKED]")
    ]
    sanitized = text
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized

# 3. Hierarchical Code Parser
# ------------------------------------------------------------------
@traceable
def parse_python_file_to_nodes(file_content: str, filename: str):
    """
    Parses Python source code to extract individual functions and classes as discrete child chunks.
    """
    child_chunks = []
    try:
        tree = ast.parse(file_content)
        lines = file_content.splitlines()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                end_line = getattr(node, "end_lineno", start_line + 5)
                if end_line > len(lines):
                    end_line = len(lines)
                
                node_code = "\n".join(lines[start_line:end_line])
                node_type = "class" if isinstance(node, ast.ClassDef) else "function"
                
                child_chunks.append({
                    "text": f"[{node_type.upper()}] {node.name} in {filename}:\n{node_code}",
                    "name": node.name,
                    "type": node_type
                })
    except SyntaxError:
        # Fallback for non-python or syntax issues: treat first part of file as a fallback chunk
        child_chunks.append({
            "text": file_content[:1000],
            "name": "raw_content",
            "type": "raw"
        })
    except Exception:
        child_chunks.append({
            "text": file_content[:1000],
            "name": "raw_content",
            "type": "raw"
        })
        
    return child_chunks

def get_or_create_index(index_name: str):
    """Checks for or creates a Serverless Pinecone Index."""
    index_exists = index_name in pc.list_indexes().names()
    
    if not index_exists:
        print(f"Creating Pinecone Index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index '{index_name}' created successfully.")
        return pc.Index(index_name), True
    else:
        print(f"Index '{index_name}' already exists.")
        return pc.Index(index_name), False

def get_vectorstore(index_name: str):
    """Returns the cached or newly initialized LangChain PineconeVectorStore."""
    if index_name not in vectorstores:
        index, _ = get_or_create_index(index_name)
        vectorstores[index_name] = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
    return vectorstores[index_name]

# 4. Zip File Processor & Ingestion Pipeline
# ------------------------------------------------------------------
# ... Keep your existing imports and setup intact ...

# Batch config constants to stay safely under limits
BATCH_SIZE = 25  # Lower this if chunks are incredibly dense/large
DELAY_SECONDS = 3.0  # Time to wait between batch uploads

@traceable
def ingest_codebase_zip(zip_path: str, index_name: str):
    """
    Extracts nested files from standard or folder-nested zip archives, 
    stores raw text inside SQLite database, and uploads child document 
    embeddings to Pinecone in controlled rate-limited batches.
    """
    index, is_new = get_or_create_index(index_name)
    
    if not is_new:
        return {
            "message": f"Index '{index_name}' already exists. Use a unique index name.",
            "chunks": 0,
            "files": 0,
            "status": "skipped"
        }
    
    vectorstore = get_vectorstore(index_name)
    child_docs_to_upload = []
    files_processed = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for file_info in archive.infolist():
                if file_info.is_dir() or file_info.filename.startswith('__MACOSX'):
                    continue
                
                filename = file_info.filename
                if '/.' in filename or filename.startswith('.'):
                    continue
                    
                if filename.endswith(('.py', '.js', '.ts')):
                    with archive.open(filename) as file:
                        try:
                            raw_content = file.read().decode('utf-8')
                            safe_content = sanitize_text(raw_content)
                        except UnicodeDecodeError:
                            continue 
                        
                        save_to_db(index_name, filename, safe_content)
                        files_processed += 1
                        
                        extracted_nodes = parse_python_file_to_nodes(safe_content, filename)
                        
                        for node in extracted_nodes:
                            child_metadata = {
                                "source_file": filename,
                                "entity_name": node["name"],
                                "entity_type": node["type"],
                                "parent_id": filename  
                            }
                            
                            child_docs_to_upload.append(
                                Document(page_content=node["text"], metadata=child_metadata)
                            )

        # --- REVISED BATCHED INGESTION LOOP ---
        total_chunks = len(child_docs_to_upload)
        if total_chunks > 0:
            print(f"Total chunks generated: {total_chunks}. Uploading in batches to stay under rate limit...")
            
            for i in range(0, total_chunks, BATCH_SIZE):
                batch = child_docs_to_upload[i:i + BATCH_SIZE]
                print(f"Uploading batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")
                
                try:
                    vectorstore.add_documents(batch)
                except Exception as batch_error:
                    # Capture specific rate limit error and try an extra backoff wait period
                    if "429" in str(batch_error) or "limit" in str(batch_error).lower():
                        print("⚠️ Hit rate limit! Pausing execution for 15 seconds to cool down...")
                        time.sleep(15)
                        vectorstore.add_documents(batch)  # Retry once after cooldown
                    else:
                        raise batch_error
                
                # Small pause between normal batches to pace token spending
                time.sleep(DELAY_SECONDS)
                
            return {
                "message": f"Successfully indexed '{index_name}'!",
                "chunks": total_chunks,
                "files": files_processed,
                "status": "success"
            }
        else:
            return {
                "message": "No compatible files found inside zip.",
                "chunks": 0,
                "files": 0,
                "status": "no_files"
            }
            
    except Exception as e:
        return {
            "message": f"Error running zip extraction pipeline: {e}",
            "chunks": 0,
            "files": 0,
            "status": "error"
        }
# 5. Metadata Reference-Retriever
# ------------------------------------------------------------------
@traceable
def metadata_parent_retriever(query: str, index_name: str, k: int = 5):
    """
    Queries Pinecone for child matches, extracts referenced file paths, 
    fetches raw full texts from SQLite, and deduplicates contexts for the LLM.
    """
    try:
        vectorstore = get_vectorstore(index_name)
        results = vectorstore.similarity_search(query, k=k)
        
        seen_parent_ids = set()
        parent_contexts = []
        
        for doc in results:
            # Get the pointer path
            parent_id = doc.metadata.get("parent_id")
            
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                
                # Fetch full contents from SQLite
                parent_text = get_from_db(index_name, parent_id)
                if parent_text:
                    parent_contexts.append(f"--- FILE: {parent_id} ---\n{parent_text}")
                
        return "\n\n".join(parent_contexts)
    except Exception as e:
        print(f"Retriever error: {e}")
        return ""

# 6. RAG Chain Definitions
# ------------------------------------------------------------------
prompt_template = """You are an expert developer assistant. Answer the user's question about the codebase using ONLY the provided code files.

Context (Relevant Source Code Files):
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def run_rag_chain(query: str, index_name: str, k: int = 5) -> str:
    """Executes the reference lookup and runs Gemini model generation."""
    # Retrieve deduplicated parent files using the SQLite references
    context = metadata_parent_retriever(query, index_name, k=k)
    
    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain.invoke({"context": context, "question": query})

# 7. FastAPI Endpoints
# ------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Codebase RAG API is active", "status": "active"}

@app.post("/upload", response_model=UploadResponse)
async def upload_codebase(
    file: UploadFile = File(...),
    index_name: str = Form(...)
):
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip archives are allowed")
    
    if not index_name or not index_name.strip():
        raise HTTPException(status_code=400, detail="Index name cannot be empty")
    
    index_name = re.sub(r'[^a-zA-Z0-9-]', '-', index_name.strip()).lower()
    temp_zip_path = f"temp_{file.filename}"
    
    try:
        content = await file.read()
        with open(temp_zip_path, 'wb') as f:
            f.write(content)
        
        result = ingest_codebase_zip(temp_zip_path, index_name)
        return UploadResponse(
            message=result["message"],
            index_name=index_name,
            chunks_uploaded=result["chunks"],
            files_processed=result["files"],
            status=result["status"]
        )
    finally:
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        index_name = re.sub(r'[^a-zA-Z0-9-]', '-', request.index_name.strip()).lower()
        
        if index_name not in pc.list_indexes().names():
            raise HTTPException(
                status_code=404, 
                detail=f"Index '{index_name}' not found. Please upload/create this index first."
            )
        
        response = run_rag_chain(request.query, index_name, k=request.k)
        
        return ChatResponse(
            answer=response,
            query=request.query,
            index_name=index_name
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing chat chain: {str(e)}")

@app.get("/indices")
async def list_indices():
    try:
        indices = pc.list_indexes().names()
        return {"indices": list(indices), "count": len(indices)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/index/{index_name}")
async def delete_index(index_name: str):
    try:
        index_name = index_name.lower()
        if index_name in pc.list_indexes().names():
            # Clear vector DB index
            pc.delete_index(index_name)
            
            # Clear metadata records from SQLite DB
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM file_content WHERE index_name=?", (index_name,))
                
            if index_name in vectorstores:
                del vectorstores[index_name]
                
            return {"message": f"Index '{index_name}' and its local database content deleted successfully", "status": "success"}
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 8. Run standard Server
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)