import os
import zipfile
import ast
import sqlite3
import time
import re
import uuid
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
app = FastAPI(title="Codebase RAG API", description="API for highly optimized hybrid codebase indexing and querying")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend origin
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
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

# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 2. Setup Pinecone, Sparse Encoding, and AI Components
# ------------------------------------------------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
dimension = 1024  # Required dimension size for multilingual-e5-large

embeddings = PineconeEmbeddings(model="multilingual-e5-large")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)

# Global dictionary to store instantiated PineconeVectorStores
vectorstores = {}

import hashlib

def encode_sparse_text(text: str):
    """
    A stable, deterministic feature hashing sparse encoder optimized for source code.
    Uses MD5 to ensure indices match perfectly across server restarts.
    """
    tokens = re.findall(r'\w+', text.lower())
    if not tokens:
        return {"indices": [0], "values": [0.0]}
        
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
        
    total = len(tokens)
    indices = []
    values = []
    
    for token, count in counts.items():
        # Use MD5 hash instead of Python's built-in non-deterministic hash()
        hf = hashlib.md5(token.encode('utf-8')).hexdigest()
        idx = int(hf[:8], 16) % (2**31 - 1) 
        indices.append(idx)
        values.append(float(count / total))
        
    sorted_pairs = sorted(zip(indices, values))
    return {
        "indices": [p[0] for p in sorted_pairs],
        "values": [p[1] for p in sorted_pairs]
    }

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

# ------------------------------------------------------------------
# 3. Hierarchical Code Parser & Index Syncing
# ------------------------------------------------------------------
@traceable
def parse_python_file_to_nodes(file_content: str, filename: str):
    """
    Parses source code files into individual functions/classes.
    Supports Python AST parsing and regex-based fallback for JS/TS architectures.
    """
    child_chunks = []
    
    # Handle Python files natively using AST
    if filename.endswith('.py'):
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno - 1
                    end_line = getattr(node, "end_lineno", start_line + 5)
                    if end_line > len(lines): end_line = len(lines)
                    
                    node_code = "\n".join(lines[start_line:end_line])
                    node_type = "class" if isinstance(node, ast.ClassDef) else "function"
                    child_chunks.append({
                        "text": f"[{node_type.upper()}] {node.name} in {filename}:\n{node_code}",
                        "name": node.name,
                        "type": node_type
                    })
            if child_chunks: return child_chunks
        except Exception:
            pass # Fall back to regex if AST fails

    # Generic Regex Parser for JS, TS, and Python fallbacks
    # Captures function declarations, arrow functions, classes, and methods
    pattern = r'(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|class\s+(\w+)|(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)'
    lines = file_content.splitlines()
    matches = list(re.finditer(pattern, file_content))
    
    if matches:
        for i, match in enumerate(matches):
            name = match.group(1) or match.group(2) or match.group(3)
            start_pos = match.start()
            # Chunk until the next match or max out at 1500 characters
            end_pos = matches[i+1].start() if i + 1 < len(matches) else start_pos + 1500
            node_code = file_content[start_pos:end_pos].strip()
            
            node_type = "class" if "class " in match.group(0) else "function"
            child_chunks.append({
                "text": f"[{node_type.upper()}] {name} in {filename}:\n{node_code}",
                "name": name,
                "type": node_type
            })
    
    # Total fallback if no distinct structures are matched
    if not child_chunks:
        for chunk_idx, i in enumerate(range(0, len(file_content), 1200)):
            child_chunks.append({
                "text": f"[RAW CHUNK {chunk_idx}] in {filename}:\n{file_content[i:i+1200]}",
                "name": "raw_content",
                "type": "raw"
            })
            
    return child_chunks
def get_or_create_index(index_name: str):
    """Checks for or creates a Serverless Pinecone Index with dotproduct metric for Hybrid Search."""
    index_exists = index_name in pc.list_indexes().names()
    
    if not index_exists:
        print(f"Creating Pinecone Index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            # CHANGE THIS FROM "cosine" TO "dotproduct"
            metric="dotproduct", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f"Index '{index_name}' created successfully.")
    
    target_index = pc.Index(index_name)
    
    if index_name not in vectorstores:
        vectorstores[index_name] = PineconeVectorStore(
            index=target_index,
            embedding=embeddings,
            text_key="text"
        )
        
    return target_index, not index_exists
def get_vectorstore(index_name: str):
    """Returns the cached or newly initialized LangChain PineconeVectorStore."""
    if index_name not in vectorstores:
        get_or_create_index(index_name)
    return vectorstores[index_name]

# ------------------------------------------------------------------
# 4. Hybrid Zip File Processor & Ingestion Pipeline
# ------------------------------------------------------------------
BATCH_SIZE = 20  
DELAY_SECONDS = 3.0  

@traceable
def ingest_codebase_zip(zip_path: str, index_name: str):
    """Extracts files, runs text/sparse pipelines, and upserts hybrid vectors in rate-limited batches."""
    index, is_new = get_or_create_index(index_name)
    
    if not is_new:
        return {
            "message": f"Index '{index_name}' already exists. Use a unique index name.",
            "chunks": 0,
            "files": 0,
            "status": "skipped"
        }
    
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
                                "text": node["text"], 
                                "source_file": filename,
                                "entity_name": node["name"],
                                "entity_type": node["type"],
                                "parent_id": filename  
                            }
                            child_docs_to_upload.append(child_metadata)

        total_chunks = len(child_docs_to_upload)
        if total_chunks > 0:
            print(f"Total chunks generated: {total_chunks}. Uploading hybrid vectors...")
            
            for i in range(0, total_chunks, BATCH_SIZE):
                batch = child_docs_to_upload[i:i + BATCH_SIZE]
                print(f"Uploading batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)...")
                
                batch_texts = [doc["text"] for doc in batch]
                vs = get_vectorstore(index_name)
                dense_embeddings = vs.embeddings.embed_documents(batch_texts)
                
                vectors_to_upsert = []
                for j, doc_metadata in enumerate(batch):
                    sparse_vector = encode_sparse_text(doc_metadata["text"])
                    
                    vectors_to_upsert.append({
                        "id": str(uuid.uuid4()),
                        "values": dense_embeddings[j],
                        "sparse_values": sparse_vector,
                        "metadata": doc_metadata
                    })
                
                try:
                    index.upsert(vectors=vectors_to_upsert)
                except Exception as batch_error:
                    if "429" in str(batch_error) or "limit" in str(batch_error).lower():
                        print("⚠️ Hit rate limit! Pausing execution for 15 seconds...")
                        time.sleep(15)
                        index.upsert(vectors=vectors_to_upsert)  
                    else:
                        raise batch_error
                
                time.sleep(DELAY_SECONDS)
                
            return {
                "message": f"Successfully indexed '{index_name}' with Hybrid search!",
                "chunks": total_chunks,
                "files": files_processed,
                "status": "success"
            }
        else:
            return {"message": "No compatible files found.", "chunks": 0, "files": 0, "status": "no_files"}
            
    except Exception as e:
        return {"message": f"Pipeline failure: {e}", "chunks": 0, "files": 0, "status": "error"}

# ------------------------------------------------------------------
# 5. Hybrid Metadata Reference-Retriever
# ------------------------------------------------------------------
@traceable
def metadata_parent_retriever(query: str, index_name: str, k: int = 5, alpha: float = 0.7):
    """Queries Pinecone using an alpha-blended Hybrid Dense/Sparse matrix payload."""
    try:
        index, _ = get_or_create_index(index_name)
        vs = get_vectorstore(index_name)
        
        raw_dense = vs.embeddings.embed_query(query)
        raw_sparse = encode_sparse_text(query)
        
        scaled_dense = [v * alpha for v in raw_dense]
        scaled_sparse = {
            "indices": raw_sparse["indices"],
            "values": [v * (1.0 - alpha) for v in raw_sparse["values"]]
        }
        
        response = index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=k,
            include_metadata=True
        )
        
        seen_parent_ids = set()
        parent_contexts = []
        
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            parent_id = metadata.get("parent_id")
            
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                
                parent_text = get_from_db(index_name, parent_id)
                if parent_text:
                    parent_contexts.append(f"--- FILE: {parent_id} ---\n{parent_text}")
                
        return "\n\n".join(parent_contexts)
    except Exception as e:
        print(f"Hybrid Retriever error: {e}")
        return ""

# ------------------------------------------------------------------
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
    context = metadata_parent_retriever(query, index_name, k=k)
    
    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain.invoke({"context": context, "question": query})

# ------------------------------------------------------------------
# 7. FastAPI Endpoints
# ------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Codebase RAG API is active", "status": "active"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

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
            pc.delete_index(index_name)
            
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM file_content WHERE index_name=?", (index_name,))
                
            if index_name in vectorstores:
                del vectorstores[index_name]
                
            return {"message": f"Index '{index_name}' and contents deleted successfully", "status": "success"}
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)